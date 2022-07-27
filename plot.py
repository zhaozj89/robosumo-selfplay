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


def parse_log(log_path, element):
    with open(log_path, 'r') as f:
        logs = f.readlines()
    result = []
    for line in logs:
        if element in line:
            frac = line.split('|')[2].strip()
            frac = float(frac)
            result.append(frac)
    return np.array(result)


parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
parser.add_argument('--type', help='figure to draw', type=str)
parser.add_argument('--opponent_mode', type=str)
args = parser.parse_args()
opponent_mode = args.opponent_mode

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
    '''
    folders = ['log_sumo/log_random_1000M_env_64', 
               'log_sumo/log_random_1000M_env_64_wo_vclip', 
               'log_sumo/log_random_1000M_env_64_nstep_1024', 
               'log_sumo/log_random_1000M_env_64_wo_vclip_nstep_1024', 
               'log_sumo_old_wo_GAE/log_random_1000M_env_64', 
               'log_sumo_old_wo_GAE/v1_with_GAE/log_random_1000M_env_64']
    labels = ['current', 'without value clip', 'nstep=1024', 'without value clip, nstep=1024', 'without GAE', 'single seed']

    folders = ['log_sumo/log_random_1000M_env_64_wo_vclip_nstep_1024', 
               'log_sumo/log_random_1000M_env_64_nstep_1024_both', 
               'log_sumo/log_random_1000M_env_64_nstep_1024_op', 
               'log_sumo/log_random_1000M_env_64_nstep_1024_direct']
    labels = ['baseline', 'OP+OE', 'OP', 'direct']

    folders = ['test2', 'log_sumo/log_random_1000M_env_64_nstep_1024_version_gap_both']
            #    'log_sumo/log_random_1000M_env_64_nstep_1024_both', 
               #'log_sumo/log_random_1000M_env_64_nstep_1024_op', 
               #'log_sumo/log_random_1000M_env_64_nstep_1024_direct', 
            #    'test_new_adv_both', 
            #    'test_new_ratio_direct']
    labels = ['baseline', 'OP+OE', 'OP', 'direct', 'new both']
    labels = ['baseline', 'version gap']

    
    # folders = ['log_sumo/log_latest_1000M_env_64_nstep_1024', 
    #            'log_sumo/log_latest_1000M_env_64_nstep_1024_direct']
    # labels = ['baseline', 'use opponent data']
    

    plt.figure(figsize=(16, 9))
    fig, axes = plt.subplots(1, 2)
    ax1 = axes[0]
    ax2 = axes[1]
    for i, folder in enumerate(folders):
        path = os.path.join(folder, 'RoboSumo-Ant-vs-Ant-v0-0', 'eval_against_fix.pkl')
        with open(path, 'rb') as f:
            result = pickle.load(f)
        result.sort(key=lambda x: x[0])
        if '1024' in folder:
            interval = 1024
        else:
            interval = 1024
        ax1.plot([x[0] * interval * 64 for x in result], [x[1] for x in result], label=labels[i])
        ax2.plot([x[0] * interval * 64 for x in result], [2000. * x[1] + (-1000.) * x[2] + (-2000.) * x[3] for x in result], label=labels[i])
        '''
        if 'test2' in folder:
            path = os.path.join(folder, 'RoboSumo-Ant-vs-Ant-v0-0', 'eval_against_fix_stochastic.pkl')
            with open(path, 'rb') as f:
                result = pickle.load(f)
            result.sort(key=lambda x: x[0])
            if '1024' in folder:
                interval = 1024
            else:
                interval = 1024
            ax1.plot([x[0] * interval * 64 for x in result], [x[1] for x in result], label=labels[i])
            ax2.plot([x[0] * interval * 64 for x in result], [2000. * x[1] + (-1000.) * x[2] + (-2000.) * x[3] for x in result], label=labels[i] + ' stochastic')
        '''
    plt.legend()
    plt.savefig('figure/compare_robosumo_random.png')
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

if args.type == 'analysis':
    methods = ['baseline', 'OP+OE', 'OP', 'direct']
    for i, suffix in enumerate(['', '_both', '_op', '_direct']):
        frac_all = []
        for seed in [0, 1, 2]:
            log_path = f'log_volleyball/log_{opponent_mode}_100M_env_8_GAE{suffix}/SlimeVolley-v0-{seed}/log.txt'
            ppo_clip_frac = parse_log(log_path)
            frac_all.append(ppo_clip_frac)
        frac_mean = np.zeros([max(len(x) for x in frac_all)])
        for seed in [0, 1, 2]:
            r = frac_all[seed]
            frac_mean[:len(r)] = frac_mean[:len(r)] + r
        frac_mean = frac_mean / 3.
        plt.plot(frac_mean, label=methods[i])
    plt.legend()
    plt.savefig(f'figure/volleyball_{opponent_mode}_ppo_clip_frac.png')
    plt.close()

if args.type == 'volley_vtrace':
    methods = ['baseline', 'rho=1', 'rho=10', 'rho=10, lam=0.5']
    for i, suffix in enumerate(['', '_both_rho_1', '_both_rho_10', '_both_rho_10_lam_0.5']):
        frac_all = []
        log_path = f'log_volleyball/log_{opponent_mode}_100M_env_8_vtrace{suffix}/SlimeVolley-v0-1/eval_against_fix.pkl'
        with open(log_path, 'rb') as f:
            result = pickle.load(f)
        plt.plot([x[0] for x in result], [x[1] for x in result], label=methods[i])

    log_path = f'log_volleyball/log_{opponent_mode}_100M_env_8_GAE/SlimeVolley-v0-1/eval_against_fix.pkl'
    with open(log_path, 'rb') as f:
        result = pickle.load(f)
    plt.plot([x[0] for x in result], [x[1] for x in result], label='GAE')

    plt.xlim(0, 1000)
    plt.legend()
    plt.savefig(f'figure/volleyball_vtrace.png')
    plt.close()

if args.type == 'sumo_baseline':
    # folders = ['test2', 
    #            'test_baseline_vtrace_adjust_z', 
    #            'log_sumo/log_random_1000M_env_64_nstep_1024_vtrace', 
    #            'test_baseline_vtrace_lam_0.5']
    folders = ['test2', 
               'log_sumo/log_random_1000M_env_64_nstep_1024_both', 
               'log_sumo/log_random_1000M_env_64_nstep_1024_version_gap_both']
    plt.figure(figsize=(16, 9))
    for j, element in enumerate(['loss/clipfrac', 'loss/policy_entropy', 'loss/policy_loss', 'loss/value_loss', \
        'misc/explained_variance', 'eplenmean', 'eprewmean']):
        plt.subplot(2, 4, j + 1)
        for i, folder in enumerate(folders):
            log_path = f'{folder}/RoboSumo-Ant-vs-Ant-v0-0/log.txt'
            result = parse_log(log_path, element)
            plt.plot(result, label=folder)
            plt.xlim(0, 1000)
        plt.title(element)
    
    plt.subplot(2, 4, 8)
    for i, folder in enumerate(folders):
        log_path = f'{folder}/RoboSumo-Ant-vs-Ant-v0-0/eval_against_fix.pkl'
        with open(log_path, 'rb') as f:
            result = pickle.load(f)
        plt.plot([x[1] for x in result], label=folder)
        # plt.xlim(0, 2000)

    plt.legend()
    plt.savefig(f'figure/compare_sumo_GAE_both.png')
    plt.close()
