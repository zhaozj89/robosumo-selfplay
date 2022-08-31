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
            seed=None, gamma=0.99, fix_opponent_path='robosumo/robosumo/policy_zoo/assets/ant/mlp/agent-params-v3.npy', 
            actor_critic=td3_core.mlp_actor_critic, ac_kwargs=dict(),
            epochs=100, replay_size=int(1e6), 
            polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
            update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
            noise_clip=0.5, policy_delay=2, max_ep_len=1000, save_freq=1
            ):

    tf.set_random_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    checkdir = osp.join(logger.get_dir(), 'checkpoints')

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

    start_time = time.time()
    all_o, ep_ret, ep_len = env.reset(), 0, 0
    o, o_oppo = all_o[:, 0, :], all_o[:, 1, :]
    steps_per_epoch = total_timesteps // epochs

    # initialize the opponent as a copy of the initial agent
    # Get state_space and action_space
    ob_space = env.observation_space[0]
    ac_space = env.action_space[0]

    fix_opponent = MLPPolicy(scope='policy1', reuse=False,
                        ob_space=ob_space,
                        ac_space=ac_space,
                        hiddens=[64, 64], normalize=True)

    if opponent_mode == 'fix':
        opponent_params = load_params(fix_opponent_path)
        set_from_flat(fix_opponent.get_variables(), opponent_params)
    else:
        fix_opponent.load(osp.join(checkdir, '00000'))

    oppo_policy = build_policy(env, fix_opponent, num_hidden=64, activation=tf.nn.relu, value_network='copy')

    def opponent_get_action(o, deterministic=False):
        act, _, _, _ = oppo_policy.step(o, deterministic=deterministic)
        return act

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
        a_oppo = opponent_get_action(o_oppo)

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
                if use_opponent_data is None:
                    batch = ego_replay_buffer.sample_batch(batch_size)
                else:
                    ego_batch = ego_replay_buffer.sample_batch(batch_size//2)
                    oppo_batch = oppo_replay_buffer.sample_batch(batch_size//2)
                    batch = np.concatenate((ego_batch, oppo_batch), dim=0)
                
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

    return pi, env

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



