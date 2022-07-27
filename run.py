import argparse
import datetime
import sys
import robosumo
import os
import pickle
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import gym
import multiprocessing
import os.path as osp
import tensorflow as tf
from baselines import logger
from defaults import get_default_params
from sumo_env import SumoEnv
from subproc_vec_env import SubprocVecEnv
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env import VecFrameStack
from baselines.common.tf_util import get_session
from baselines.common import set_global_seeds
from baselines.bench import Monitor

from slimevolleygym.slimevolley import SlimeVolleyEnv


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def make_env_from_id(env_id, logger_dir, mpi_rank, subrank, seed, prefix):
    env = gym.make(env_id)
    if 'RoboSumo' in env_id:
        for agent in env.agents:
            agent._adjust_z = -0.5
        env = SumoEnv(env, allow_early_resets=True, file_prefix=prefix)
    print (seed)
    env.seed(seed)
    env = Monitor(env, logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)), allow_early_resets=True)
    print (env)
    return env

# def make_vec_env(env_id, num_env, seed,
#                  wrapper_kwargs=None,
#                  env_kwargs=None,
#                  start_index=0,
#                  reward_scale=1.0,
#                  flatten_dict_observations=True,
#                  gamestate=None,
#                  initializer=None,
#                  force_dummy=False):
#     """
#     Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
#     """
#     wrapper_kwargs = wrapper_kwargs or {}
#     env_kwargs = env_kwargs or {}
#     mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
#     seed = seed + 10000 * mpi_rank if seed is not None else None
#     logger_dir = logger.get_dir()
#     def make_thunk(rank, initializer=None):
#         return lambda: make_env_from_id(
#             env_id=env_id,
#             # env_type=env_type,
#             mpi_rank=mpi_rank,
#             subrank=rank,
#             seed=seed,
#             # reward_scale=reward_scale,
#             # gamestate=gamestate,
#             # flatten_dict_observations=flatten_dict_observations,
#             # wrapper_kwargs=wrapper_kwargs,
#             # env_kwargs=env_kwargs,
#             logger_dir=logger_dir,
#             # initializer=initializer
#         )

#     set_global_seeds(seed)
#     if not force_dummy and num_env > 1:
#         return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
#     else:
#         return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])

def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    print ('args.num_env', args.num_env, ncpu)
    nenv = args.num_env or ncpu
    print('num of env: ' + str(nenv))

    seed = args.seed
    env_id = args.env

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    # frame_stack_size = 4
    # env = make_vec_env(env_id, nenv, seed)
    # env = VecFrameStack(env, frame_stack_size)

    env = SubprocVecEnv([lambda i=i: make_env_from_id(env_id, args.log_path, 0, i, seed + i if seed is not None else None, "")
                         for i in range(nenv)])
    return env


def train(args, extra_args):
    # assert args.env[:8] == 'RoboSumo'

    env_id = args.env
    pg_method = args.pg
    print (args)

    # build a temporary environment to get number of agents
    temp_env = gym.make(env_id)
    nagent = len(temp_env.agents)
    temp_env.close()

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    alg_kwargs = get_default_params(env_id, pg_method)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = 'mlp'

    print('Training PPO2 on {} with arguments \n{}'.format(env_id, alg_kwargs))
    with open(os.path.join(args.log_path, 'config.pkl'), 'wb') as f:
        pickle.dump([args, alg_kwargs], f)

    if pg_method == 'ppo':
        from alg import learn
    else:
        from alg_ac import learn

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        nagent=nagent,
        **alg_kwargs
    )

    return model, env


def main(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='RoboSumo-Ant-vs-Ant-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp')
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel.', default=1, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=".", type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default="./logs", type=str)
    parser.add_argument('--suffix', help='', default="default", type=str)
    parser.add_argument('--pg', type=str, default='ppo')

    # implicit args: opponent_mode use_opponent_data
    args, unknown_args = parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if args.log_path is not None:
        # args.log_path = osp.join(args.log_path,
        #                          args.env + '-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
        args.log_path = osp.join(args.log_path, args.env + '-' + args.suffix)
    os.system('rm -r %s'%(args.log_path))
    configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)

    # if args.save_path is not None:
    #     save_path = osp.expanduser(args.save_path)
    # else:
    #     save_path = osp.join(args.save_path, 'model')
    #print('Saving final model to', 'model')
    #model.save('model.ckpt')

    env.close()
    return model


if __name__ == '__main__':
    main(sys.argv)
