import tensorflow as tf


def get_default_params(task, pg_method):
    if 'RoboSumo' in task:
        if pg_method == 'ppo':
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
        else:
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
    