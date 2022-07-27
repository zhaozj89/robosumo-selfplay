import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from policies import build_policy
from runner import Runner
import tensorflow as tf
import copy

from robosumo.policy_zoo.utils import load_params, set_from_flat
from robosumo.policy_zoo import MLPPolicy

import matplotlib.pyplot as plt


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, network, env, total_timesteps, opponent_mode='ours', use_opponent_data=None, eval_env=None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4, vf_coef=0.5,
          max_grad_norm=0.5, gamma=0.99, lam=0.95, log_interval=10, 
          save_interval=1, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None,
          nagent=1, anneal_bound=500, fix_opponent_path='robosumo/robosumo/policy_zoo/assets/ant/mlp/agent-params-v3.npy', **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    nagent: int                       number of agents in an environment

    anneal_bound: int                 the number of iterations it takes for dense reward anneal to 0

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)
    print ('grad norm', max_grad_norm)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space[0]
    ac_space = env.action_space[0]

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from model import ActorCriticModel
        model_fn = ActorCriticModel

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=None, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                     model_scope='model_%d' % 0)

    models = [model]
    checkdir = osp.join(logger.get_dir(), 'checkpoints')
    model.save(osp.join(checkdir, '00000'))
    for i in range(1, nagent):
        models.append(
            model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=None, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, trainable=False,
                     model_scope='model_%d' % i))
    writer = tf.summary.FileWriter(logger.get_dir(), tf.get_default_session().graph)

    model_util = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=None, nbatch_train=nbatch_train,
                        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, trainable=False,
                        model_scope='model_util')

    # plot the weights distribution in checkpoint models
    '''
    for update in range(180, 200, 1):
        model_util.load(osp.join(checkdir, '%.5i'%update))
        variables = tf.trainable_variables(scope=model_util.scope)
        sess = model_util.sess
        ps = sess.run(variables)
        ps = np.concatenate([x.ravel() for x in ps])
        print (ps)
        print (np.isnan(ps).sum())
        plt.figure()
        plt.hist(ps, bins=100)
        plt.savefig(osp.join(logger.get_dir(), 'fig', 'weight_%d.png' %(update)))
        plt.close()
    return model, env
    '''

    if load_path is not None:
        for i in range(nagent):
            models[i].load(load_path)

    # Instantiate the runner object
    runner = Runner(env=env, models=models, nsteps=nsteps, nagent=nagent, gamma=gamma, lam=lam, anneal_bound=anneal_bound)
    if eval_env is not None:
        eval_runner = Runner(env=eval_env, models=models, nsteps=nsteps, nagent=nagent, gamma=gamma, lam=lam)
    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    # number of iterations
    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        if update % log_interval == 0:
            print('Iteration: %d/%d' % (update, nupdates))
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)

        # Set opponents' model
        if update == 1:
            # initialize the opponent as a copy of the initial agent
            if opponent_mode == 'fix':
                fix_opponent = MLPPolicy(scope='policy1', reuse=False,
                                    ob_space=ob_space,
                                    ac_space=ac_space,
                                    hiddens=[64, 64], normalize=True)
                opponent_params = load_params(fix_opponent_path)
                set_from_flat(fix_opponent.get_variables(), opponent_params)
                print (fix_opponent.get_variables())
                print (runner.models[1])
                opponent_policy = build_policy(env, fix_opponent, num_hidden=64, activation=tf.nn.relu, value_network='copy')
                runner.models[1] = model_fn(policy=opponent_policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=None, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, trainable=False,
                    model_scope='model_1')
            else:
                runner.models[1].load(osp.join(checkdir, '00000'))
            if update % log_interval == 0:
                logger.info('Stepping environment...Compete with v0')
        else:
            # different environment get different opponent model
            # all parallel environments get same opponent model
            #old_versions = [round(np.random.uniform(1, update - 1)) for _ in range(nagent - 1)]
            #old_model_paths = [osp.join(checkdir, '%.5i' % old_id) for old_id in old_versions]
            old_model_paths = [osp.join(checkdir, f) for f in os.listdir(checkdir)]
            old_model_paths.sort()
            #assert(nagent==2, 'ONLY support two agents training')
            if opponent_mode=='random':
                idx = np.random.choice(len(old_model_paths), 1)[0]
                runner.models[1].load(old_model_paths[idx])
            elif opponent_mode=='latest':
                idx = update - 1
                runner.models[1].load(old_model_paths[-1])

            if update % log_interval == 0:
                logger.info('Stepping environment...Compete with version %d' %(idx))

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, rewards, opponent_obs, opponent_actions, states, epinfos, opponent_neglogpacs = runner.run(update)
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_rewards, _, _, \
            eval_states, eval_epinfos = eval_runner.run()

        # # transfer opponent data
        # clip_ratio = 5.
        # # compute off-policy, off-env ratio
        # #off_policy_ratio = np.exp(opponent_neglogpacs - models[0].act_model.action_probability(obs[1], given_action=actions[1]))
        # off_policy_ratio = np.exp(opponent_neglogpacs - neglogpacs[1])
        # off_policy_clip_frac = (off_policy_ratio > clip_ratio).mean()
        # off_policy_ratio = np.clip(off_policy_ratio, 0., clip_ratio)
        # off_policy_ratio[np.isnan(off_policy_ratio)] = clip_ratio
        # #off_policy_ratio = models[0].act_model.action_probability(obs[1], actions[1]) / np.exp(-neglogpacs[1])
        # off_env_ratio = np.exp(neglogpacs[0] - models[1].act_model.action_probability(obs[0], given_action=actions[0]))
        # off_env_clip_frac = (off_env_ratio > clip_ratio).mean()
        # off_env_ratio = np.clip(off_env_ratio, 0., clip_ratio)
        # off_env_ratio[np.isnan(off_env_ratio)] = clip_ratio
        # #off_env_ratio = models[1].act_model.action_probability(obs[0], actions[0]) / np.exp(-neglogpacs[0])
        # total_ratio = off_policy_ratio * off_env_ratio
        # total_clip_frac = (total_ratio > clip_ratio).mean()
        # total_ratio = np.clip(total_ratio, 0., clip_ratio)
        # total_ratio[np.isnan(total_ratio)] = clip_ratio

        # discard the opponent samples where the action probability is too low
        neglogp_threshold = 50.
        #usable_index = np.where(neglogpacs[1] < neglogp_threshold)[0]
        usable_index = np.where(models[0].act_model.action_probability(obs[1], given_action=actions[1]) < neglogp_threshold)[0]
        #logger.info(f'use {usable_index.shape[0]} of {neglogpacs[1].shape[0]} opponent samples')

        # # plot ratios for visualization
        # plt.figure(figsize=(16, 9))
        # plt.subplot(2, 2, 1)
        # plt.hist(np.clip(np.log(off_policy_ratio), -5., 5.), bins=100)
        # plt.ticklabel_format(useOffset=False)
        # plt.title('off-policy ratio (log scale): %.2f%% clipped' %(off_policy_clip_frac * 100.))
        # plt.subplot(2, 2, 2)
        # plt.hist(np.clip(np.log(off_env_ratio), -5., 5.), bins=100)
        # plt.ticklabel_format(useOffset=False)
        # plt.title('off-env ratio (log scale): %.2f%% clipped' %(off_env_clip_frac * 100.))
        # plt.subplot(2, 2, 3)
        # plt.hist(np.clip(np.log(total_ratio), -5., 5.), bins=100)
        # plt.ticklabel_format(useOffset=False)
        # plt.title('off-policy-env ratio (log scale): %.2f%% clipped' %(total_clip_frac * 100.))
        # plt.subplot(2, 2, 4)
        # plt.hist(np.clip(neglogpacs[1].ravel(), -neglogp_threshold, neglogp_threshold), bins=100)
        # plt.ticklabel_format(useOffset=False)
        # plt.title('-neglogp of \pi_1(a^2|o^2)')
        # if update == 1:
        #     plt.suptitle('opponent version: 0')
        # else:
        #     plt.suptitle(f'opponent version: {idx}')
        # os.makedirs(osp.join(logger.get_dir(), 'fig'), exist_ok=True)
        # plt.savefig(osp.join(logger.get_dir(), 'fig', 'ratio_%d.png' %(update)))

        # logger.info('neglogp statistics check: max, min, inf, nan')
        # logger.info(f'{neglogpacs[1].max()}, {neglogpacs[1].min()}, {np.isinf(neglogpacs[1]).sum()}, {np.isnan(neglogpacs[1]).sum()}')
        # logger.info('opponent data value function check: max, min')
        # logger.info(f'{values[1].max()}, {values[1].min()}')

        if use_opponent_data is None:
            obs, returns, masks, actions, values, neglogpacs, rewards = \
                list(map(lambda x: x[0], (obs, returns, masks, actions, values, neglogpacs, rewards)))
        else:
            obs, returns, masks, actions, values, neglogpacs, rewards = \
                list(map(lambda x: np.concatenate([x[0], x[1, usable_index]], axis=0), (obs, returns, masks, actions, values, neglogpacs, rewards)))

        if use_opponent_data is None:
            weights = np.ones(nbatch, dtype=np.float32)
        elif use_opponent_data == 'direct':
            weights = np.ones(obs.shape[0], dtype=np.float32)
        elif use_opponent_data == 'off_policy':
            weights = np.concatenate([np.ones(nbatch, dtype=np.float32), off_policy_ratio[usable_index]])
        elif use_opponent_data == 'both':
            weights = np.concatenate([np.ones(nbatch, dtype=np.float32), total_ratio[usable_index]])

        if update % log_interval == 0:
            logger.info('Done.')

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        slices = (arr for arr in (obs, returns, masks, actions, values, neglogpacs, rewards, weights))
        temp_out = model.train(lrnow, *slices)
        writer.add_summary(temp_out[-1], (update - 1))
        mblossvals.append(temp_out[:-1])

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('epdenserewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            # if eval_env is not None:
            #     logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
            #     logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)

            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)

    writer.close()
    return model


# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



