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
import pickle
import utils.td3_core as td3_core

from robosumo.policy_zoo.utils import load_params, set_from_flat
from robosumo.policy_zoo import MLPPolicy

import matplotlib.pyplot as plt


def constfn(val):
    def f(_):
        return val
    return f

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def learn(*, env, total_timesteps, opponent_mode='ours', use_opponent_data=None, 
            seed=None, gamma=0.99, save_interval=1, load_path=None, model_fn=None, 
            nagent=1, anneal_bound=500, fix_opponent_path='robosumo/robosumo/policy_zoo/assets/ant/mlp/agent-params-v3.npy', 
            actor_critic=td3_core.mlp_actor_critic, ac_kwargs=dict(),
            epochs=100, replay_size=int(1e6), 
            polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
            update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
            noise_clip=0.5, policy_delay=2, max_ep_len=1000, 
            save_freq=1, **network_kwargs,
            ):

    tf.set_random_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = td3_core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    # Target policy network
    with tf.variable_scope('target'):
        pi_targ, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)
    
    # Opponent's policy network
    with tf.variable_scope('opponent'):
        pi_targ, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Target Q networks
    with tf.variable_scope('target', reuse=True):

        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        # Target Q-values, using action from target policy
        _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)

    # Experience buffer
    ego_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    oppo_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(td3_core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q functions, using Clipped Double-Q targets
    min_q_targ = tf.minimum(q1_targ, q2_targ)
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_targ)

    # TD3 losses
    pi_loss = -tf.reduce_mean(q1_pi)
    q1_loss = tf.reduce_mean((q1-backup)**2)
    q2_loss = tf.reduce_mean((q2-backup)**2)
    q_loss = q1_loss + q2_loss

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=td3_core.get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=td3_core.get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(td3_core.get_vars('main'), td3_core.get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(td3_core.get_vars('main'), td3_core.get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def opponent_get_action(o, noise_scale):
        pass

    start_time = time.time()
    all_o, ep_ret, ep_len = env.reset(), 0, 0
    o, o_oppo = all_o[:, 0, :], all_o[:, 1, :]
    steps_per_epoch = total_timesteps // epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_timesteps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # get opponent action
        a_oppo = opponent_get_action(o_oppo, act_noise)

        # Step the env
        # all_actions shape: num_env * nagents * action_dim
        all_actions = np.array([[a, a_oppo]])
        all_o2, r, d, _ = env.step(all_actions)
        o2, o2_oppo = all_o2[:, 0, :], all_o2[:, 1, :]
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        ego_replay_buffer.store(o, a, r, o2, d)
        oppo_replay_buffer.store(o_oppo, a_oppo, r, o2_oppo, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2
        o_oppo = o2_oppo

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            all_o, ep_ret, ep_len = env.reset(), 0, 0
            o, o_oppo = all_o[:, 0, :], all_o[:, 1, :]

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = ego_replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }
                q_step_ops = [q_loss, q1, q2, train_q_op]
                outs = sess.run(q_step_ops, feed_dict)
                logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                if j % policy_delay == 0:
                    # Delayed policy update
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

def learn(*, network, env, total_timesteps, opponent_mode='ours', use_opponent_data=None, eval_env=None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4, vf_coef=0.5,
          max_grad_norm=0.5, gamma=0.99, lam=0.95, rho_bar=1., c_bar=1., log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=1, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None,
          nagent=1, anneal_bound=500, fix_opponent_path='robosumo/robosumo/policy_zoo/assets/ant/mlp/agent-params-v3.npy', 
          vgap=None, kl_threshold=0.01, **network_kwargs):

    set_global_seeds(seed)
    print ('grad norm', max_grad_norm)
    print ('max vgap', vgap)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space[0]
    ac_space = env.action_space[0]

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from model import PPOModel
        model_fn = PPOModel

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

    if load_path is not None:
        for i in range(nagent):
            models[i].load(load_path)

    # Instantiate the runner object
    runner = Runner(env=env, models=models, nsteps=nsteps, nagent=nagent, 
                    gamma=gamma, lam=lam, rho_bar=rho_bar, c_bar=c_bar, anneal_bound=anneal_bound)
    if eval_env is not None:
        eval_runner = Runner(env=eval_env, models=models, nsteps=nsteps, nagent=nagent, gamma=gamma, lam=lam)
    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    off_policy_ratio_mean, off_env_ratio_mean, total_ratio_mean = [], [], []
    off_policy_ratio_clip_frac, off_env_ratio_clip_frac, total_ratio_clip_frac = [], [], []
    ppo_clip_frac, approxkl = [], []
    version_gap = []
    useful_ratio = []

    # number of iterations
    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        print('Iteration: %d/%d' % (update, nupdates))
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

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
                version_gap.append(0)
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
                idx = np.random.choice(update, 1)[0]
                runner.models[1].load(old_model_paths[idx])
                version_gap.append(update - 1 - idx)
            elif opponent_mode=='latest':
                idx = update - 1
                runner.models[1].load(old_model_paths[-1])
            elif opponent_mode=='ours':
                action_prob = runner.models[1].act_model.action_probability(opponent_obs, opponent_actions)
                if len(old_model_paths) > 30:
                    subsample_idx = np.random.choice(len(old_model_paths), 30, replace=False)
                    subsample_idx = np.sort(subsample_idx)
                else:
                    subsample_idx = np.arange(len(old_model_paths))
                RD_list = []
                for i in subsample_idx:
                    model_util.load(old_model_paths[i])
                    new_action_prob = model_util.act_model.action_probability(opponent_obs, opponent_actions)
                    ratio_divergence = (np.abs(new_action_prob / action_prob - 1.)).mean()
                    RD_list.append(ratio_divergence)
                print (RD_list)
                RD_list = np.array(RD_list)
                RD_list = RD_list / RD_list.sum()
                idx = subsample_idx[np.random.choice(len(RD_list), 1, p=RD_list)[0]]
                runner.models[1].load(old_model_paths[idx])

            if update % log_interval == 0:
                logger.info('Stepping environment...Compete with version %d' %(idx))

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, rewards, opponent_neglogpacs, \
            opponent_obs, opponent_actions, states, epinfos, \
            off_policy_ratio, off_env_ratio, total_ratio = runner.run(update)
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_rewards, _, _, \
            eval_states, eval_epinfos = eval_runner.run()

        # # transfer opponent data
        clip_ratio = rho_bar
        # compute off-policy, off-env ratio
        #off_policy_ratio = np.exp(opponent_neglogpacs - models[0].act_model.action_probability(obs[1], given_action=actions[1]))
        # off_policy_ratio = np.exp(opponent_neglogpacs - neglogpacs[1])
        off_policy_ratio[np.isnan(off_policy_ratio)] = clip_ratio
        off_policy_ratio_mean.append(off_policy_ratio.mean())
        off_policy_clip_frac = (off_policy_ratio > clip_ratio).mean()
        off_policy_ratio = np.clip(off_policy_ratio, 0., clip_ratio)
        off_policy_ratio_clip_frac.append(off_policy_clip_frac)
        #off_policy_ratio = models[0].act_model.action_probability(obs[1], actions[1]) / np.exp(-neglogpacs[1])
        # off_env_ratio = np.exp(neglogpacs[0] - models[1].act_model.action_probability(obs[0], given_action=actions[0]))
        off_env_ratio[np.isnan(off_env_ratio)] = clip_ratio
        off_env_ratio_mean.append(off_env_ratio.mean())
        off_env_clip_frac = (off_env_ratio > clip_ratio).mean()
        off_env_ratio = np.clip(off_env_ratio, 0., clip_ratio)
        off_env_ratio_clip_frac.append(off_env_clip_frac)
        #off_env_ratio = models[1].act_model.action_probability(obs[0], actions[0]) / np.exp(-neglogpacs[0])
        # total_ratio = off_policy_ratio * off_env_ratio
        total_ratio[np.isnan(total_ratio)] = clip_ratio
        total_ratio_mean.append(total_ratio.mean())
        total_clip_frac = (total_ratio > clip_ratio).mean()
        total_ratio = np.clip(total_ratio, 0., clip_ratio)
        total_ratio_clip_frac.append(total_clip_frac)

        # discard the opponent samples where the action probability is too low
        # for computational stability
        neglogp_threshold = 10000.
        # neglogp_threshold = 50.
        usable_index = np.where(neglogpacs[1] < neglogp_threshold)[0]
        # usable_index = np.where(models[0].act_model.action_probability(obs[1], given_action=actions[1]) < neglogp_threshold)[0]
        useless_ratio = 1. - len(usable_index) / len(neglogpacs[1])
        useful_ratio.append(1. - useless_ratio)

        # if use_opponent_data is None or (vgap is not None and version_gap[-1] > vgap):
        if use_opponent_data is None:
            obs, returns, masks, actions, values, neglogpacs, rewards = \
                list(map(lambda x: x[0], (obs, returns, masks, actions, values, neglogpacs, rewards)))
        else:
            logger.info('use opponent data')
            logger.info(f'{usable_index.shape[0]} of {neglogpacs[1].shape[0]} opponent samples usable')
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
        # Index of each element of batch_size
        # Create the indices array
        update_sample_num = obs.shape[0]
        inds = np.arange(update_sample_num)

        ratio_all_epoch = []
        early_stop = False
        for epoch in range(noptepochs):
            ratio = np.zeros_like(inds, dtype=np.float32)
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            # the minibatch number is doubled if we use opponent data
            for ii, start in enumerate(range(0, update_sample_num, nbatch_train)):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs, rewards, weights))
                temp_out = model.train(lrnow, cliprangenow, *slices)
                writer.add_summary(temp_out[-1], (update - 1) * noptepochs * nminibatches + epoch * nminibatches + ii)
                #for i in range(5):
                #    logger.info(f'{model.loss_names[i]}: {temp_out[i].mean()}')
                mblossvals.append(temp_out[:-2])
                batch_ratio = temp_out[-2]
                ratio[mbinds] = batch_ratio
                kl = temp_out[3]
                if kl_threshold is not None:
                    if kl > kl_threshold * 1.5:
                        early_stop = True
                        break
            ratio_all_epoch.append(ratio)
            if early_stop:
                print (f'early stop at epoch {epoch} iter {ii} due to large approxkl')
                break

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        ppo_clip_frac.append(lossvals[-1])
        approxkl.append(lossvals[-2])
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
        
        if update % 100 == 0 or update == 1:
            with open(osp.join(logger.get_dir(), 'ratio_summary.pkl'), 'wb') as f:
                pickle.dump([version_gap, off_policy_ratio_mean, off_env_ratio_clip_frac, 
                    off_env_ratio_mean, off_env_ratio_clip_frac, 
                    total_ratio_mean, total_clip_frac, 
                    ppo_clip_frac, approxkl], f)
            size = 1.
            plt.figure(figsize=(16, 9))
            
            plt.subplot(2, 4, 1)
            plt.scatter(version_gap, off_policy_ratio_mean, s=size)
            plt.title('off policy ratio')
            plt.ylabel('ratio mean')
            plt.subplot(2, 4, 5)
            plt.scatter(version_gap, off_policy_ratio_clip_frac, s=size)
            plt.ylabel('ratio clip frac')
            plt.xlabel('version gap')

            plt.subplot(2, 4, 2)
            plt.scatter(version_gap, off_env_ratio_mean, s=size)
            plt.title('off env ratio')
            plt.subplot(2, 4, 6)
            plt.scatter(version_gap, off_env_ratio_clip_frac, s=size)
            plt.xlabel('version gap')

            plt.subplot(2, 4, 3)
            plt.scatter(version_gap, total_ratio_mean, s=size)
            plt.title('total ratio')
            plt.subplot(2, 4, 7)
            plt.scatter(version_gap, total_ratio_clip_frac, s=size)
            plt.xlabel('version gap')

            plt.subplot(3, 4, 4)
            plt.scatter(version_gap, ppo_clip_frac, s=size)
            plt.title('ppo clip frac')
            plt.subplot(3, 4, 8)
            plt.scatter(version_gap, approxkl, s=size)
            plt.title('approxkl')
            plt.subplot(3, 4, 12)
            plt.scatter(version_gap, useful_ratio, s=size)
            plt.title('useful opponent data')

            plt.savefig(osp.join(logger.get_dir(), 'ratio_summary.png'))
            plt.close()

    writer.close()
    return model


# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



