# python3 compare_history_version.py --p1 ours --p2 random --trials 50

import tensorflow as tf
import os.path as osp
from policies import build_policy
import gym
import robosumo
from tqdm import tqdm
import time
import os
import numpy as np
import pickle
import argparse


def evaluate(model, env, args):
    obs = env.reset()
    dones = [False, False]
    reward = None
    ep_id = 0
    total_reward = [0., 0.]
    total_scores = [0, 0]

    while True:
        action1, _, _, _ = model[0].step(obs[0])
        action2, _, _, _ = model[1].step(obs[1])
        obs, reward, dones, infos = env.step([action1[0], action2[0]])

        for i in range(2):
            total_reward[i] += reward[i]
        if dones[0]:
            #print('-' * 5 + 'Episode %d ' % (ep_id + 1) + '-' * 5)
            ep_id += 1
            draw = True
            for i in range(2):
                if 'winner' in infos[i]:
                    draw = False
                    total_scores[i] += 1
                    #print("Winner: Agent {}, Scores: {}, Total Episodes: {}".format(i, total_scores, ep_id))
            #if draw:
                #print("Match tied: Scores: {}, Total Episodes: {}".format(total_scores, ep_id))
            obs = env.reset()
            total_reward = [0. for _ in range(2)]
            if ep_id == args.trials:
                break

    return total_scores



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
    parser.add_argument('--p1', help='choice of opponent strategy for p1', type=str, default="ours")
    parser.add_argument('--p2', help='choice of opponent strategy for p2', type=str, default="ours")
    #parser.add_argument('--path', help='model path', type=str, default=None)
    #parser.add_argument('--max_version', type=int, default=1e10)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()


    p1_model_path = 'logs_' + args.p1 + '/RoboSumo-Ant-vs-Ant-v0-0/checkpoints'
    p1_model_list = [f for f in os.listdir(p1_model_path) if f != '00000']
    p1_model_list.sort()

    p2_model_path = 'logs_' + args.p2 + '/RoboSumo-Ant-vs-Ant-v0-0/checkpoints'
    p2_model_list = [f for f in os.listdir(p2_model_path) if f != '00000']
    p2_model_list.sort()

    env = gym.make('RoboSumo-Ant-vs-Ant-v0')
    env.num_envs = 1

    for agent in env.agents:
        agent._adjust_z = -0.5

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

    from model import Model
    model_fn = Model

    model = [model_fn(policy=policy[0], ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=None,
                      nsteps=None, ent_coef=None, vf_coef=None, max_grad_norm=None, trainable=False, model_scope="model_0"),
             model_fn(policy=policy[1], ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=None,
                      nsteps=None, ent_coef=None, vf_coef=None, max_grad_norm=None, trainable=False, model_scope="model_1")]

    win_rate = []
    for i in range(len(p1_model_list)):
        print (i)
        model[0].load(p1_model_path + '/' + p1_model_list[i])
        model[1].load(p2_model_path + '/' + p2_model_list[i])
        scores = evaluate(model, env, args)
        win_rate.append(scores[0] / args.trials)
    with open('compare_history_version_%s_against_%s.pkl' %(args.p1, args.p2), 'wb') as f:
        pickle.dump(win_rate, f)

