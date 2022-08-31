# python3 eval_robosumo_against_fix.py --path log_sumo/log_random_1000M_env_64/RoboSumo-Ant-vs-Ant-v0-0 --min_version 1 --max_version 2000 --trials 50 --interval 50

import tensorflow as tf
import os.path as osp
from video_recorder import VideoRecorder
from policies import build_policy
import gym
import robosumo
from tqdm import tqdm
import time
import os
import numpy as np
import zipfile
import io
import pickle

from robosumo.policy_zoo.utils import load_params, set_from_flat
# from robosumo.policy_zoo import LSTMPolicy, MLPPolicy
from subproc_vec_env import SubprocVecEnv

from model import PPOModel, ActorCriticModel

import argparse
from baselines.common.tf_util import get_session

import slimevolleygym.slimevolley
from baselines.bench import Monitor

from tensorflow.contrib import layers

from robosumo.policy_zoo.utils import *
from sumo_env import SumoEnv


class Policy(object):
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        raise NotImplementedError


class MLPPolicy(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens,
                 normalize=False,
                 reuse=False):
        self.recurrent = False
        self.normalized = normalize

        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            self.observation_ph = tf.placeholder(
                tf.float32, [None] + list(ob_space.shape), name="observation")
            self.taken_action_ph = tf.placeholder(
                tf.float32, [None, ac_space.shape[0]], name="taken_action")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(
                    scope="obsfilter", shape=ob_space.shape)

            # Observation filtering
            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            # Value
            last_out = obz
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(
                    dense(last_out, hid_size, "vffc%i" % (i + 1)))
            self.vpredz = dense(last_out, 1, "vffinal")[:, 0]

            self.vpred = self.vpredz
            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean

            # Policy
            last_out = obz
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(
                    dense(last_out, hid_size, "polfc%i" % (i + 1)))
            mean = dense(last_out, ac_space.shape[0], "polfinal")
            logstd = tf.get_variable(
                name="logstd",
                shape=[1, ac_space.shape[0]],
                initializer=tf.zeros_initializer())

            self.pd = DiagonalGaussian(mean, logstd)
            self.sampled_action = switch(
                self.stochastic_ph, self.pd.sample(), self.pd.mode())

    def act(self, observation, stochastic=True):
        outputs = [self.sampled_action, self.vpred]
        feed_dict = {
            self.observation_ph: observation,
            self.stochastic_ph: stochastic,
        }
        a, v = tf.get_default_session().run(outputs, feed_dict)
        return a, {'vpred': v}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)


def make_env_from_id(env_id, logger_dir, mpi_rank, subrank, seed, prefix):
    env = gym.make(env_id)
    for agent in env.agents:
        agent._adjust_z = -0.5
    env = SumoEnv(env, allow_early_resets=True, file_prefix=prefix)
    env.seed(seed)
    #env = Monitor(env, logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)), allow_early_resets=True)
    return env


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
    parser.add_argument('--label', help='choice of opponent strategy', type=str, default="ours")
    parser.add_argument('--path', help='model path', type=str, default=None)
    parser.add_argument('--min_version', type=int, default=1)
    parser.add_argument('--max_version', type=int, default=1e10)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--task', type=str, default='robosumo')
    parser.add_argument('--num_env', type=int, default=32)
    parser.add_argument('--pg', type=str, default='ppo')
    parser.add_argument('--opponent_version', type=int, default=3)
    args = parser.parse_args()

    # configure
    path = args.path
    # ID_length = min(len(list(os.listdir(path + '/checkpoints'))), args.max_version) - 1
    ID_length = args.max_version
    current_id = max(0, args.min_version)
    round_total = args.trials
    num_env = args.num_env

    # record
    ep_id = 0
    # total_reward = [0., 0.]
    # total_scores = [0, 0]
    dones = [False, False]
    reward = None

    # make an environment
    env_id = 'RoboSumo-Ant-vs-Ant-v0'
    env = SubprocVecEnv([lambda seed=seed: make_env_from_id(env_id, 'test', 0, 0, seed, '') for seed in range(num_env)])
    #env.render('human')

    #env = VideoRecorder(env, osp.join(path, "videos"), record_video_trigger=lambda x: True, video_length=5000)

    policy = build_policy(env, 'mlp', num_hidden=64, activation=tf.nn.relu, value_network='copy')
    ob_space = env.observation_space[0]
    ac_space = env.action_space[0]
    print (ob_space)

    # agent
    if args.pg == 'ppo':
        model_fn = PPOModel
    else:
        model_fn = ActorCriticModel
    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=None, nbatch_train=None,
                    nsteps=None, ent_coef=None, vf_coef=None, max_grad_norm=None, trainable=False, model_scope="model_0")

    model_path = path + '/checkpoints/%.5i' % current_id
    model.load(model_path)

    # fixed opponent
    low = -np.inf * np.ones(120)
    high = np.inf * np.ones(120)
    opponent_ob_space = gym.spaces.Box(low, high)
    opponent_dir = "robosumo/robosumo/policy_zoo/assets/ant/mlp/agent-params-v%d.npy"%(args.opponent_version)
    opponent_policy = MLPPolicy(scope='policy1', reuse=False,
                                ob_space=opponent_ob_space,
                                ac_space=ac_space,
                                hiddens=[64, 64], normalize=True)
    opponent_params = load_params(opponent_dir)
    set_from_flat(opponent_policy.get_variables(), opponent_params)

    '''
    # load existing evaluation results
    try:
        with open(os.path.join(path, 'eval_against_fix.pkl'), 'rb') as f:
            s_win_rate = pickle.load(f)
        evaluated_id = [x[0] for x in s_win_rate]
    except:
        s_win_rate = []
        evaluated_id = []
    print (evaluated_id)
    evaluated_id = []
    '''
    s_win_rate = []

    obs = env.reset()
    not_done = np.ones(num_env)
    win_rate, draw_rate, lose_rate = 0, 0, 0
    round_num = 0
    step_num = 0
    while True:
        #env.render('human')
        action1, _, _, _ = model.step(obs[:, 0, :], deterministic=True)
        # action2 = np.stack([opponent_policy.act(stochastic=False, observation=obs[i, 1, :])[0] for i in range(num_env)], axis=0)
        action2, _ = opponent_policy.act(stochastic=False, observation=obs[:, 1, :-1])
        all_actions = np.stack([action1, action2], axis=1)
        obs, reward, dones, infos = env.step(all_actions)

        finish_idx = np.where(dones[:, 0])[0]
        round_num += len(finish_idx)
        for idx in finish_idx:
            if 'winner' in infos[idx][0]:
                win_rate += 1
            elif 'winner' in infos[idx][1]:
                lose_rate += 1
            else:
                draw_rate += 1
            
            # if idx == 0:
            #     print (obs[0,0,:10])
        
        step_num += 1

        if round_num >= round_total:
            print('-' * 5 + 'Episode %d win: %.2f, draw: %.2f' %(current_id, win_rate/round_num, draw_rate/round_num) + '-' * 5)
            print ('run for %d steps'%(step_num))
            s_win_rate.append([current_id, win_rate / round_num, draw_rate / round_num, lose_rate / round_num])
            win_rate, draw_rate, lose_rate = 0, 0, 0

            round_num = 0
            step_num = 0

            current_id += args.interval
            if current_id>ID_length:
                break
            
            model_path = path + '/checkpoints/%.5i' % current_id
            try:
                model.load(model_path)
            except:
                break

            # env = SubprocVecEnv([lambda seed=seed: make_env_from_id(env_id, 'test', 0, 0, seed, '') for seed in range(num_env)])
            obs = env.reset()

        # finish_idx = np.where(dones[:, 0])[0]
        # for idx in finish_idx:
        #     if not_done[idx] == 1:
        #         if 'winner' in infos[idx][0]:
        #             win_rate += 1
        #         elif 'winner' in infos[idx][1]:
        #             lose_rate += 1
        #         else:
        #             draw_rate += 1
        #         not_done[idx] = 0

        # if not_done.sum() == 0:
        #     print('-' * 5 + 'Episode {} win: {}, draw: {}'.format(current_id, win_rate/num_env, draw_rate/num_env) + '-' * 5)
        #     s_win_rate.append([current_id, win_rate / num_env, draw_rate / num_env, lose_rate / num_env])
        #     win_rate, draw_rate, lose_rate = 0, 0, 0

        #     '''
        #     while (1):
        #         current_id += args.interval
        #         if current_id not in evaluated_id:
        #             break
        #     '''
        #     current_id += args.interval
        #     if current_id>ID_length:
        #         break
        #     model_path = path + '/checkpoints/%.5i' % current_id
        #     try:
        #         model.load(model_path)
        #     except:
        #         break
        #     not_done = np.ones(num_env)
        #     obs = env.reset()

    with open(os.path.join(path, 'eval_against_fix_v%d.pkl'%(args.opponent_version)), 'wb') as f:
        pickle.dump(s_win_rate, f)
