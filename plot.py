from baselines.common import plot_util as pu
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import pickle


parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
parser.add_argument('--type', help='figure to draw', type=str)
args = parser.parse_args()

if args.type == 'train_reward':

    labels = ['ours', 'random', 'latest']
    folders = ['logs_ours', 'logs_random', 'logs_latest']
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

    labels = ['ours', 'random', 'latest']
    plt.figure()
    for label in labels:
        win_rate = pd.read_csv('eval_against_fix_%s.csv' %(label))
        plt.plot(win_rate, label=label)
    plt.legend()
    plt.savefig('eval_against_fix.png')
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