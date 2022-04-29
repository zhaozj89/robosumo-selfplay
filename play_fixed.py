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


path = ['RoboSumo-Ant-vs-Ant-v0-2022-04-17-11-59-54-559389',
        'RoboSumo-Ant-vs-Ant-v0-2022-04-17-11-59-54-559389']
ID = [122, 122]
length = 5000
model_path = [path[0] + '/checkpoints/%.5i' % ID[0], path[1] + '/checkpoints/%.5i' % ID[1]]

env = gym.make('RoboSumo-Ant-vs-Ant-v0')
env.num_envs = 1

for agent in env.agents:
    agent._adjust_z = -0.5

# env = VideoRecorder(env, osp.join(path, "videos-%d" % mid), record_video_trigger=lambda x: True, video_length=length)

policy = [build_policy(env, 'mlp', num_hidden=64, activation=tf.nn.relu, value_network='copy'),
          build_policy(env, 'mlp', num_hidden=64, activation=tf.nn.relu, value_network='copy')]
ob_space = env.observation_space[0]
ac_space = env.action_space[0]

from robosumo.policy_zoo.utils import load_params, set_from_flat
from robosumo.policy_zoo import LSTMPolicy, MLPPolicy

tf_config = tf.ConfigProto(
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1)
sess = tf.Session(config=tf_config)
sess.__enter__()
sess.run(tf.variables_initializer(tf.global_variables()))

opponent_dir = os.path.join("robosumo/robosumo/policy_zoo/assets/ant/mlp/agent-params-v3.npy")
opponent_policy = MLPPolicy(scope='policy1', reuse=False,
                            ob_space=ob_space,
                            ac_space=ac_space,
                            hiddens=[64, 64], normalize=True)
opponent_params = load_params(opponent_dir)
set_from_flat(opponent_policy.get_variables(), opponent_params)


from model import Model
model_fn = Model

model = [model_fn(policy=policy[0], ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=None,
                  nsteps=None, ent_coef=None, vf_coef=None, max_grad_norm=None, trainable=False, model_scope="model_0"),
         model_fn(policy=policy[1], ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=None,
                  nsteps=None, ent_coef=None, vf_coef=None, max_grad_norm=None, trainable=False, model_scope="model_1")]
model[0].load(model_path[0])
model[1].load(model_path[1])

ep_id = 0
total_reward = [0., 0.]
total_scores = [0, 0]

# env.render('human')
obs = env.reset()
dones = [False, False]
reward = None

for step in range(length):
    # env.render('human')
    # time.sleep(0.01)
    action1, _, _, _ = model[0].step(obs[0])
    action2, _ = opponent_policy.act(stochastic=True, observation=obs[1])
    obs, reward, dones, infos = env.step([action1[0], action2])
    # action2, _, _, _ = model[1].step(obs[1])
    # obs, reward, dones, infos = env.step([action1[0], action2[0]])

    for i in range(2):
        total_reward[i] += reward[i]
    if dones[0]:
        print('-' * 5 + 'Episode %d ' % (ep_id + 1) + '-' * 5)
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

