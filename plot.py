# python3 plot.py --type eval_against_fix --opponent_mode random
from baselines.common import plot_util as pu
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import pickle
import os


parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
parser.add_argument('--type', help='figure to draw', type=str)
parser.add_argument('--opponent_mode', type=str)
args = parser.parse_args()

if args.type == 'train_reward':

    labels = ['ours', 'random', 'latest']
    folders = ['logs_ours', 'logs_random', 'logs_latest']
    labels = ['ours', 'origin']
    folders = ['logs_ours_100M', 'logs_origin_100M']
    results = {}

    for label, folder in zip(labels, folders):
        results[label] = pu.load_results(folder)
        print (len(results[label]))

    plt.figure(figsize=(16, 9))
    for label in labels:
        reward = np.zeros(len(results[label][0].progress['misc/total_timesteps']))
        for r in results[label]:
            reward += r.progress.eprewmean
        reward = reward / len(results[label])
        plt.plot(results[label][0].progress['misc/total_timesteps'], reward, label=label)
    plt.legend()
    plt.savefig('train_reward.png')
    plt.close()

if args.type == 'eval_against_fix':

    #labels = ['random_env_64', 'random_env_64_direct', 'random_env_64_op', 'random_env_64_both']
    #labels = ['random_env_64', 'random_env_32', 'random_env_16']
    '''
    labels = ['volleyball_random_env_8', 'volleyball_random_env_8_both']
    plt.figure()
    for label in labels:
        win_rate = pd.read_csv('eval_against_fix_%s.csv' %(label))
        plt.plot(win_rate, label=label)
        if label == 'random_env_64_50_20':
            plt.plot([i * 20 for i in range(win_rate.shape[0])], win_rate, label=label)
        else:
            plt.plot([i * 10 for i in range(win_rate.shape[0])], win_rate, label=label)
        #plt.plot([int(label.split('_')[-1]) * (50 * i) * 8196 for i in range(len(win_rate))], win_rate, label=label)
    plt.legend()
    plt.savefig('eval_against_fix.png')
    plt.close()
    '''
    opponent_mode = args.opponent_mode
    plt.figure()
    '''
    result_without_GAE = None
    for seed in [0, 1, 2]:
        path = f'log_volleyball/log_{opponent_mode}_100M_env_8/SlimeVolley-v0-{seed}/eval_against_fix.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                result = pickle.load(f)
            result.sort(key=lambda x: x[0])
            if result_without_GAE is None:
                result_without_GAE = np.array([x[1] for x in result])
            else:
                result_without_GAE = result_without_GAE + np.array([x[1] for x in result])
    plt.plot([x[0] for x in result], result_without_GAE / 3, label='without GAE', c='r')
    '''
    methods = ['baseline', 'OP+OE', 'OP', 'direct']
    for i, suffix in enumerate(['', '_both', '_op', '_direct']):
        result_with_GAE = None
        for seed in [0, 1, 2]:
            path = f'log_volleyball/log_{opponent_mode}_100M_env_8_GAE{suffix}/SlimeVolley-v0-{seed}/eval_against_fix.pkl'
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    result = pickle.load(f)
                result.sort(key=lambda x: x[0])
                if result_with_GAE is None:
                    result_with_GAE = np.array([x[1] for x in result])
                    xticks = np.array([x[0] for x in result])
                else:
                    new_result = np.array([x[1] for x in result])
                    result_with_GAE[:len(new_result)] = result_with_GAE[:len(new_result)] + new_result
        plt.plot(xticks, result_with_GAE / 3, label=methods[i])
    plt.legend()
    plt.savefig(f'eval_against_fix_{opponent_mode}.png')
    plt.close()

if args.type == 'compare_history_version':

    files = ['compare_history_version_ours_against_random.pkl', 
             'compare_history_version_ours_against_latest.pkl', 
             'compare_history_version_ours_against_ours.pkl']
    labels = ['ours vs random', 'our vs latest', 'ours vs ours']
    files = ['compare_history_version_latest_against_random.pkl', 'compare_history_version_random_against_latest.pkl']
    labels = ['latest vs random', 'random vs latest']
    plt.figure(figsize=(16, 9))
    for data, label in zip(files, labels):
        with open(data, 'rb') as f:
            win_rate = pickle.load(f)
        plt.plot(win_rate, label=label)
    plt.xlabel('agent version')
    plt.ylabel('win rate')
    plt.legend()
    plt.savefig('compare_history_version.png')
    plt.close()