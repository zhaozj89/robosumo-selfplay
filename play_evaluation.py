# python3 play_evaluation.py --label ours --path logs/RoboSumo-Ant-vs-Ant-v0-0 --min_version 1 --max_version 1 --trials 10 --interval 10

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
import pandas as pd

from robosumo.policy_zoo.utils import load_params, set_from_flat
from robosumo.policy_zoo import LSTMPolicy, MLPPolicy

from model import Model
model_fn = Model

import argparse
from baselines.common.tf_util import get_session


parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
parser.add_argument('--label', help='choice of opponent strategy', type=str, default="ours")
parser.add_argument('--path', help='model path', type=str, default=None)
parser.add_argument('--min_version', type=int, default=1)
parser.add_argument('--max_version', type=int, default=1e10)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--interval', type=int, default=10)
args = parser.parse_args()

# configure
path = args.path
ID_length = min(len(list(os.listdir(path + '/checkpoints'))), args.max_version) - 1
current_id = max(1, args.min_version)
round_total = args.trials

# record
ep_id = 0
# total_reward = [0., 0.]
# total_scores = [0, 0]
dones = [False, False]
reward = None
s_win_rate = pd.Series([0])

# make an environment
env = gym.make('RoboSumo-Ant-vs-Ant-v0')
env.num_envs = 1

#env = VideoRecorder(env, osp.join(path, "videos"), record_video_trigger=lambda x: True, video_length=5000)

for agent in env.agents:
    agent._adjust_z = -0.5

policy = build_policy(env, 'mlp', num_hidden=64, activation=tf.nn.relu, value_network='copy')
ob_space = env.observation_space[0]
ac_space = env.action_space[0]

# agent
model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=None,
                  nsteps=None, ent_coef=None, vf_coef=None, max_grad_norm=None, trainable=False, model_scope="model_0")

model_path = path + '/checkpoints/%.5i' % current_id
model.load(model_path)

# fixed opponent
'''
tf_config = tf.ConfigProto(
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1)
sess = tf.Session(config=tf_config)
sess.__enter__()
#sess.run(tf.variables_initializer(tf.global_variables()))
'''

opponent_dir = "robosumo/robosumo/policy_zoo/assets/ant/mlp/agent-params-v3.npy"
opponent_policy = MLPPolicy(scope='policy1', reuse=False,
                            ob_space=ob_space,
                            ac_space=ac_space,
                            hiddens=[64, 64], normalize=True)
opponent_params = load_params(opponent_dir)
set_from_flat(opponent_policy.get_variables(), opponent_params)

'''
print (model)
sess = model.sess
variables = tf.trainable_variables(scope=model.scope)
ps = sess.run(variables)
print (ps)
'''

#env.render('human')
obs = env.reset()

round_nb = 0
win_rate = 0
while True:
    #env.render('human')
    action1, _, _, _ = model.step(obs[0])
    action2, _ = opponent_policy.act(stochastic=True, observation=obs[1])
    obs, reward, dones, infos = env.step([action1[0], action2])

    if dones[0]:
        round_nb += 1
        if 'winner' in infos[0]:
            win_rate += 1

        if round_nb == round_total:
            round_nb = 0
            print('-' * 5 + 'Episode {} winning rate: {}'.format(current_id, win_rate/round_total) + '-' * 5)
            s_win_rate.loc[s_win_rate.index.max()+1] = win_rate/round_total
            win_rate = 0

            current_id += args.interval
            if current_id>ID_length:
                break
            model_path = path + '/checkpoints/%.5i' % current_id
            model.load(model_path)

        obs = env.reset()

s_win_rate.to_csv('eval_against_fix_%s.csv' %(args.label), index = False)
