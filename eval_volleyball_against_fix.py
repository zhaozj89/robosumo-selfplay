# python3 eval_volleyball_against_fix.py --path log_volleyball/log_latest_100M_env_8_GAE_op/SlimeVolley-v0-0 --min_version 1 --max_version 4000 --trials 50 --interval 50

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

from subproc_vec_env import SubprocVecEnv

from model import Model
model_fn = Model

import argparse
from baselines.common.tf_util import get_session

import slimevolleygym.slimevolley
from baselines.bench import Monitor


def make_env_from_id(env_id, logger_dir, mpi_rank, subrank, seed, prefix):
    env = gym.make(env_id)
    env.seed(seed)
    #env = Monitor(env, logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)), allow_early_resets=True)
    return env


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
    parser.add_argument('--path', help='model path', type=str, default=None)
    parser.add_argument('--min_version', type=int, default=1)
    parser.add_argument('--max_version', type=int, default=1e10)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--task', type=str, default='robosumo')
    parser.add_argument('--num_env', type=int, default=32)
    args = parser.parse_args()

    # configure
    path = args.path
    ID_length = min(len(list(os.listdir(path + '/checkpoints'))), args.max_version) - 1
    current_id = max(1, args.min_version)
    round_total = args.trials
    num_env = round_total
    env_id = 'SlimeVolley-v0'

    # record
    ep_id = 0
    # total_reward = [0., 0.]
    # total_scores = [0, 0]
    dones = [False, False]
    reward = None
    s_win_rate = []

    #env = [lambda s=seed: make_env_from_id(env_id, 'test', 0, 0, s, '') for seed in range(num_env)]
    env = SubprocVecEnv([lambda s=seed: make_env_from_id(env_id, 'test', 0, 0, s, '') for seed in range(num_env)])
    #env.render('human')

    #env = VideoRecorder(env, osp.join(path, "videos"), record_video_trigger=lambda x: True, video_length=5000)

    policy = build_policy(env, 'mlp', num_hidden=64, activation=tf.nn.relu, value_network='copy')
    ob_space = env.observation_space[0]
    ac_space = env.action_space[0]

    # agent
    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=None, nbatch_train=None,
                    nsteps=None, ent_coef=None, vf_coef=None, max_grad_norm=None, trainable=False, model_scope="model_0")

    model_path = path + '/checkpoints/%.5i' % current_id
    model.load(model_path)

    # fixed opponent
    opponent_policy = [slimevolleygym.slimevolley.BaselinePolicy() for _ in range(num_env)]

    obs = env.reset()
    win_rate = 0
    not_done = np.ones(num_env)
    while True:
        #env.render('human')
        action1, _, _, _ = model.step(obs[:, 0, :], deterministic=True)
        action2 = np.stack([opponent_policy[i].step(obs[i, 1, :]) for i in range(num_env)], axis=0)
        all_actions = np.stack([action1, action2], axis=1)
        obs, reward, dones, infos = env.step(all_actions)
        
        finish_idx = np.where(dones[:, 0])[0]
        #print (finish_idx)
        for idx in finish_idx:
            if not_done[idx] == 1:
                win_rate += (infos[idx][0]['ale.lives'] - infos[idx][0]['ale.otherLives'])
                #print (infos[idx][0]['ale.lives'] - infos[idx][0]['ale.otherLives'])
                not_done[idx] = 0
            #opponent_policy[idx].reset()
        
        if not_done.sum() == 0:
            print('-' * 5 + 'Episode {} avg score: {}'.format(current_id, win_rate/num_env) + '-' * 5)
            s_win_rate.append([current_id, win_rate / num_env])
            win_rate = 0

            current_id += args.interval
            if current_id > ID_length:
                break
            model_path = path + '/checkpoints/%.5i' % current_id
            model.load(model_path)
            not_done = np.ones(num_env)
            obs = env.reset()
            for op in opponent_policy:
                op.reset()
            

    with open(os.path.join(path, 'eval_against_fix.pkl'), 'wb') as f:
        pickle.dump(s_win_rate, f)
