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
from robosumo.policy_zoo import LSTMPolicy, MLPPolicy
from subproc_vec_env import SubprocVecEnv

from model import PPOModel, ActorCriticModel

import argparse
from baselines.common.tf_util import get_session

import slimevolleygym.slimevolley
from baselines.bench import Monitor


def make_env_from_id(env_id, logger_dir, mpi_rank, subrank, seed, prefix):
    env = gym.make(env_id)
    for agent in env.agents:
        agent._adjust_z = -0.5
    #env = SumoEnv(env, allow_early_resets=True, file_prefix=prefix)
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
    args = parser.parse_args()

    # configure
    path = args.path
    # ID_length = min(len(list(os.listdir(path + '/checkpoints'))), args.max_version) - 1
    ID_length = args.max_version
    current_id = max(0, args.min_version)
    round_total = args.trials
    num_env = round_total

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
    opponent_dir = "robosumo/robosumo/policy_zoo/assets/ant/mlp/agent-params-v3.npy"
    opponent_policy = MLPPolicy(scope='policy1', reuse=False,
                                ob_space=ob_space,
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
    while True:
        #env.render('human')
        action1, _, _, _ = model.step(obs[:, 0, :], deterministic=True)
        action2 = np.stack([opponent_policy.act(stochastic=False, observation=obs[i, 1, :])[0] for i in range(num_env)], axis=0)
        all_actions = np.stack([action1, action2], axis=1)
        obs, reward, dones, infos = env.step(all_actions)

        finish_idx = np.where(dones[:, 0])[0]
        for idx in finish_idx:
            if not_done[idx] == 1:
                if 'winner' in infos[idx][0]:
                    win_rate += 1
                elif 'winner' in infos[idx][1]:
                    lose_rate += 1
                else:
                    draw_rate += 1
                not_done[idx] = 0

        if not_done.sum() == 0:
            print('-' * 5 + 'Episode {} win: {}, draw: {}'.format(current_id, win_rate/num_env, draw_rate/num_env) + '-' * 5)
            s_win_rate.append([current_id, win_rate / num_env, draw_rate / num_env, lose_rate / num_env])
            win_rate, draw_rate, lose_rate = 0, 0, 0

            '''
            while (1):
                current_id += args.interval
                if current_id not in evaluated_id:
                    break
            '''
            current_id += args.interval
            if current_id>ID_length:
                break
            model_path = path + '/checkpoints/%.5i' % current_id
            try:
                model.load(model_path)
            except:
                break
            not_done = np.ones(num_env)
            obs = env.reset()

    with open(os.path.join(path, 'eval_against_fix.pkl'), 'wb') as f:
        pickle.dump(s_win_rate, f)
