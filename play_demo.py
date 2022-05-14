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
from gym.wrappers import Monitor
from robosumo.policy_zoo import LSTMPolicy, MLPPolicy
from robosumo.policy_zoo.utils import load_params, set_from_flat
from baselines.common.tf_util import get_session

model_path_1 = 'logs_random/RoboSumo-Ant-vs-Ant-v0-0/checkpoints/00300'
#model_path_2 = 'logs_latest/RoboSumo-Ant-vs-Ant-v0-0/checkpoints/00300'
model_path_2 = "robosumo/robosumo/policy_zoo/assets/ant/mlp/agent-params-v3.npy"
output_path = 'video_random_300_against_fix_test'
mode = 'fix'
length = 5000

env = gym.make('RoboSumo-Ant-vs-Ant-v0')
env.num_envs = 1

for agent in env.agents:
    agent._adjust_z = -0.5

#env = VideoRecorder(env, output_path, record_video_trigger=lambda x: True, video_length=length)
#env = Monitor(env, output_path, force=True)

policy = [build_policy(env, 'mlp', num_hidden=64, activation=tf.nn.relu, value_network='copy'),
          build_policy(env, 'mlp', num_hidden=64, activation=tf.nn.relu, value_network='copy')]
ob_space = env.observation_space[0]
ac_space = env.action_space[0]

from model import Model
model_fn = Model

model = [model_fn(policy=policy[0], ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=None,
                  nsteps=None, ent_coef=None, vf_coef=None, max_grad_norm=None, trainable=False, model_scope="model_0"),
         model_fn(policy=policy[1], ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=None,
                  nsteps=None, ent_coef=None, vf_coef=None, max_grad_norm=None, trainable=False, model_scope="model_1")]
model[0].load(model_path_1)

if mode == 'checkpoint':
    model[1].load(model_path_2)
else:
    model[1] = MLPPolicy(scope='policy1', reuse=False,
                         ob_space=ob_space,
                         ac_space=ac_space,
                         hiddens=[64, 64], normalize=True)
    opponent_params = load_params(model_path_2)
    print ('loaded params')
    print (opponent_params[-8:])
    set_from_flat(model[1].get_variables(), opponent_params)

ep_id = 0
total_reward = [0., 0.]
total_scores = [0, 0]
total_shaping_reward = [0., 0.]

sess = get_session()
print (sess.run(model[1].get_variables())[-1])

#env.render('human')
obs = env.reset()

'''
print (model[0])
sess = model[0].sess
variables = tf.trainable_variables(scope=model[0].scope)
ps = sess.run(variables)
print (ps)
'''

for _ in range(length):
    #env.render('human')
    #time.sleep(0.01)
    action1, _, _, _ = model[0].step(obs[0])
    if mode == 'checkpoint':
        action2, _, _, _ = model[1].step(obs[1])
        obs, reward, dones, infos = env.step([action1[0], action2[0]])
    else:
        action2, _ = model[1].act(stochastic=True, observation=obs[1])
        obs, reward, dones, infos = env.step([action1[0], action2])

    for i in range(2):
        total_reward[i] += reward[i]
        total_shaping_reward[i] += infos[i]['shaping_reward']
    if dones[0]:
        print('-' * 5 + 'Episode %d ' % (ep_id + 1) + '-' * 5)
        print ('total reward: ', total_reward[0], total_reward[1])
        print ('total shaping reward: ', total_shaping_reward[0], total_shaping_reward[1])        
        ep_id += 1
        draw = True
        for i in range(2):
            if 'winner' in infos[i]:
                draw = False
                total_scores[i] += 1
                print("Winner: Agent {}, Scores: {}, Total Episodes: {}".format(i, total_scores, ep_id))
        if draw:
            print("Match tied: Scores: {}, Total Episodes: {}".format(total_scores, ep_id))
        obs = env.reset()
        total_reward = [0. for _ in range(2)]
        total_shaping_reward = [0. for _ in range(2)]

env.close()
