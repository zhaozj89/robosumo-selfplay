import tensorflow as tf
from utils import td3_core


def get_default_params(task, algo):
    if 'RoboSumo' in task:
        if algo == 'ppo':
            return dict(
                nsteps=8192, #     nbatch = nenvs * nsteps
                nminibatches=32, #     nbatch_train = nbatch // nminibatches,     nupdates = total_timesteps//nbatch
                lam=1.,
                # lam=0.95, # old lambda used for GAE
                gamma=0.995,
                rho_bar=10., 
                c_bar=1., 
                noptepochs=6,
                log_interval=1,
                save_interval=1,
                ent_coef=0.0,
                lr=1e-3,
                cliprange=0.2,
                value_network='copy',
                anneal_bound=1000,
                num_hidden=64,
                activation=tf.nn.relu,
            )
        elif algo == 'td3':
            return dict(
                actor_critic=td3_core.mlp_actor_critic, 
                steps_per_epoch=4000, 
                epochs=100, 
                replay_size=int(1e6), 
                gamma=0.99, 
                polyak=0.995, 
                pi_lr=1e-3, 
                q_lr=1e-3, 
                batch_size=100, 
                start_steps=10000, 
                update_after=1000, 
                update_every=50, 
                act_noise=0.1, 
                target_noise=0.2, 
                noise_clip=0.5, 
                policy_delay=2, 
                num_test_episodes=10, 
                max_ep_len=1000, 
                save_freq=1,
            )
        elif algo == 'ac':
            return dict(
                nsteps=5, #     nbatch = nenvs * nsteps
                lam=0.95,
                gamma=0.995,
                log_interval=1000,
                save_interval=3000,
                ent_coef=0.0,
                lr=3e-4,
                value_network='copy',
                anneal_bound=1000,
                num_hidden=64,
                activation=tf.nn.relu,
            )
        else:
            raise NotImplementedError

    elif 'SlimeVolley' in task:
        return dict(
            nsteps=4096, #     nbatch = nenvs * nsteps
            nminibatches=64, #     nbatch_train = nbatch // nminibatches,     nupdates = total_timesteps//nbatch
            lam=1.,
            # lam=0.95,
            gamma=0.99,
            rho_bar=10., 
            c_bar=1., 
            noptepochs=10,
            log_interval=1,
            save_interval=1,
            ent_coef=0.0,
            lr=3e-4,
            cliprange=0.2,
            value_network='copy', 
            num_hidden=64, 
            activation=tf.nn.relu,
        )
    