import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, models, nsteps, nagent, anneal_bound):
        self.env = env
        self.models = models
        self.nenv = nenv = self.env.num_envs
        self.nagent = nagent
        # add one dimension of time
        self.obs = np.zeros((nenv,) + (len(env.observation_space),) + env.observation_space[0].shape,
                            dtype=models[0].train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = [model.initial_state for model in self.models]
        self.dones = np.array([[False for _ in range(self.nagent)] for _ in range(self.nenv)])
        # hyperparameter from "Emergent Complexity via Multi-Agent Competition"
        self.anneal_bound = anneal_bound

    @abstractmethod
    def run(self, update):
        raise NotImplementedError

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences with 1 trainable agent and old-version agents
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, models, nsteps, nagent, gamma, lam, rho_bar, c_bar, anneal_bound=500):
        super().__init__(env=env, models=models, nsteps=nsteps, nagent=nagent, anneal_bound=anneal_bound)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.rho_bar = rho_bar
        self.c_bar = c_bar

    def run(self, update):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs = [[] for _ in range(self.nagent)]
        mb_rewards = [[] for _ in range(self.nagent)]
        mb_actions = [[] for _ in range(self.nagent)]
        mb_values = [[] for _ in range(self.nagent)]
        mb_dones = [[] for _ in range(self.nagent)]
        mb_neglogpacs = [[] for _ in range(self.nagent)]
        # store the opponent's action probability to compute IS ratio
        mb_opponent_neglogpacs = [[] for _ in range(self.nagent)]
        # store opponent's observation and action to compute action probability
        opponent_obs, opponent_actions = [], []

        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        # for _ in tqdm(range(self.nsteps)):
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            all_actions = []
            for agt in range(self.nagent):
                actions, values, self.states[agt], neglogpacs = self.models[agt].step(self.obs[:, agt, :],
                                                                                      S=self.states[agt],
                                                                                      M=self.dones[:, agt])
                mb_obs[agt].append(self.obs[:, agt, :].copy())
                mb_actions[agt].append(actions)
                mb_dones[agt].append(self.dones[:, agt])

                # mb_values[agt].append(values)
                # mb_neglogpacs[agt].append(neglogpacs)

                # if agt == 1:
                #     opponent_neglogpacs.append(neglogpacs)
                #     opponent_obs.append(self.obs[:, 1, :].copy())
                #     opponent_actions.append(actions)

                if agt == 0:
                    mb_values[agt].append(values)
                    mb_neglogpacs[agt].append(neglogpacs)
                    mb_opponent_neglogpacs[agt].append(self.models[1].act_model.action_probability(self.obs[:, agt, :], given_action=actions))
                else:
                    mb_opponent_neglogpacs[agt].append(neglogpacs)
                    
                    agent_values = self.models[0].value(self.obs[:, agt, :], S=self.states[agt], M=self.dones[:, agt])
                    agent_neglogpacs = self.models[0].act_model.action_probability(self.obs[:, agt, :], given_action=actions)
                    #agent_values, agent_neglogpacs = self.models[0].act_model.value_and_neglogp(self.obs[:, agt, :], given_action=actions)
                    mb_values[agt].append(agent_values)
                    mb_neglogpacs[agt].append(agent_neglogpacs)

                    opponent_obs.append(self.obs[:, 1, :].copy())
                    opponent_actions.append(actions)
                all_actions.append(actions)
            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            all_actions = np.stack(all_actions, axis=1)
            # all_actions shape: num_env * nagents * action_dim
            # obs shape: num_env * agent_num * ob dimension for each agent (120)
            # `dones` and `infos` shape: num_env * agent_num
            self.obs[:], rewards, self.dones, infos = self.env.step(all_actions)
            # for e in range(self.dones.shape[0]):
            #     if self.dones[e].sum() != 0:
            #         print (rewards[e])
            # for info in infos:
            #     if 'timeout' in info[0]:
            #         print ('draw!')
            #print (self.obs.shape, self.dones.shape, len(infos), len(infos[0]))
            """
            info:
                shaping_reward:
                    ctrl_reward: The l_2 penalty on the actions to prevent jittery/unnatural movements
                    move_to_opp_reward: Reward at each time step proportional to magnitude of the velocity component 
                        towards the opponent.
                    push_opp_reward: The agent got penalty at each time step proportional to exp{-d_opp} where d_opp was
                        the distance of the opponent from the center the ring
                main_reward:
                    lose_penalty: -2000
                    win_reward: 2000
                episode_info: (when itersdones is True, automatically reset the env)

            """
            # exploration curriculum: r_t = alpha * shaping_reward + (1 - alpha) * done * main_reward
            if 'shaping_reward' in infos[0][0]:
                alpha = 0
                if update <= self.anneal_bound:
                    alpha = np.linspace(1, 0, self.anneal_bound)[update - 1]
                for agt in range(self.nagent):
                    rewards = np.zeros(self.nenv)
                    for e in range(self.nenv):
                        rewards[e] = alpha * infos[e][agt]['shaping_reward'] + (1 - alpha) * infos[e][agt]['main_reward']
                        # if self.dones[e, agt] and 'timeout' in infos[e][agt]:
                            # print ('draw!')
                            # print (infos[e][agt]['main_reward'])
                            # rewards[e] += self.gamma * self.models[0].value(mb_obs[agt][-1])[e]
                        if agt == 0:
                            maybeepinfo = infos[e][0].get('episode')
                            if maybeepinfo:
                                epinfos.append(maybeepinfo)
                    mb_rewards[agt].append(rewards)
            else:
                for agt in range(self.nagent):
                    mb_rewards[agt].append(rewards[:, agt])
                    if agt == 0:
                        for e in range(self.nenv):
                            maybeepinfo = infos[e][0].get('episode')
                            if maybeepinfo:
                                epinfos.append(maybeepinfo)
        # batch of steps to batch of rollouts
        # shape: n_agents * time_step * num_env
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_opponent_neglogpacs = np.asarray(mb_opponent_neglogpacs)
        #print (mb_obs.shape, mb_rewards.shape, mb_actions.shape, mb_values.shape, mb_neglogpacs.shape, mb_dones.shape)

        opponent_obs = np.asarray(opponent_obs, dtype=self.obs.dtype)
        opponent_actions = np.asarray(opponent_actions)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        # compute importance sampling ratios
        off_policy_ratio = np.exp(mb_opponent_neglogpacs[1] - mb_neglogpacs[1])
        off_env_ratio = np.exp(mb_neglogpacs[0] - mb_opponent_neglogpacs[0])
        ratio = off_policy_ratio * off_env_ratio
        # V-trace
        for agt in range(self.nagent):
            if agt == 0:
                rho_clip = np.ones_like(ratio)
                c_clip = np.ones_like(ratio)
            else:
                rho_clip = np.clip(ratio, None, self.rho_bar)
                c_clip = np.clip(ratio, None, self.c_bar)
                # print ((rho_clip != 1).sum(), (c_clip != 1).sum())
            c_clip *= self.lam
            # last_values = self.models[agt].value(self.obs[:, agt, :], S=self.states[agt], M=self.dones[:, agt])
            last_values = self.models[0].value(self.obs[:, agt, :], S=self.states[agt], M=self.dones[:, agt])
            acc = np.zeros(ratio.shape[1])
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones[:, agt]
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[agt, t + 1]
                    nextvalues = mb_values[agt, t + 1]
                # delta = rho * [r + gamma * V(s') * (1 - done) - V(s)]
                delta = rho_clip[t] * (mb_rewards[agt, t] + self.gamma * nextvalues * nextnonterminal - mb_values[agt, t])
                acc = delta + self.gamma * nextnonterminal * c_clip[t] * acc
                mb_returns[agt, t] = mb_values[agt, t] + acc
                if t == self.nsteps - 1:
                    mb_advs[agt, t] = mb_rewards[agt, t] + self.gamma * nextnonterminal * last_values - mb_values[agt, t]
                else:
                    mb_advs[agt, t] = mb_rewards[agt, t] + self.gamma * nextnonterminal * mb_returns[agt, t + 1] - mb_values[agt, t]

        # # MC returns
        # mc_mb_returns = np.zeros_like(mb_rewards)
        # for agt in range(self.nagent):
        #     # last_values = self.models[agt].value(self.obs[:, agt, :], S=self.states[agt], M=self.dones[:, agt])
        #     last_values = self.models[0].value(self.obs[:, agt, :], S=self.states[agt], M=self.dones[:, agt])
        #     for t in reversed(range(self.nsteps)):
        #         if t == self.nsteps - 1:
        #             nextnonterminal = 1.0 - self.dones[:, agt]
        #             nextvalues = last_values
        #         else:
        #             nextnonterminal = 1.0 - mb_dones[agt, t + 1]
        #             nextvalues = mc_mb_returns[agt, t + 1]
        #         # delta = rho * [r + gamma * V(s') * (1 - done) - V(s)]
        #         mc_mb_returns[agt, t] = mb_rewards[agt, t] + self.gamma * nextvalues * nextnonterminal
        # mc_mb_advs = mc_mb_returns - mb_values

        # # compare online vtrace and MC returns
        # plt.figure()
        # for e in range(mc_mb_returns.shape[2]):
        #     plt.subplot(2, 2, e + 1)
        #     plt.plot(mc_mb_returns[0, :, e], label='MC')
        #     plt.plot(mb_returns[0, :, e], label='vtrace')
        #     print (mc_mb_returns[0, :, e] - mb_returns[0, :, e])
        #     dones = np.where(mb_dones[0, :, e])[0]
        #     for idx in dones:
        #         plt.plot([idx, idx], [mc_mb_returns[0, :, e].min(), mc_mb_returns[0, :, e].max()], '--', c='k')
        # plt.legend()
        # plt.savefig('check_returns_%d.png'%(update))
        # plt.close()

        # GAE
        # for agt in range(self.nagent):
        #     lastgaelam = 0
        #     last_values = self.models[agt].value(self.obs[:, agt, :], S=self.states[agt], M=self.dones[:, agt])
        #     # last_values = self.models[0].value(self.obs[:, agt, :], S=self.states[agt], M=self.dones[:, agt])
        #     for t in reversed(range(self.nsteps)):
        #         if t == self.nsteps - 1:
        #             nextnonterminal = 1.0 - self.dones[:, agt]
        #             nextvalues = last_values
        #         else:
        #             nextnonterminal = 1.0 - mb_dones[agt, t + 1]
        #             nextvalues = mb_values[agt, t + 1]
        #         # delta = r + gamma * V(s') * (1 - done) - V(s)
        #         delta = mb_rewards[agt, t] + self.gamma * nextvalues * nextnonterminal - mb_values[agt, t]
        #         # For simplicity, don't use GAE for now
        #         mb_advs[agt, t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        #         #mb_advs_vanilla[agt, t] = delta
        #     mb_returns[agt] = mb_advs[agt] + mb_values[agt]
        
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_rewards, mb_opponent_neglogpacs, opponent_obs, opponent_actions)),
                mb_states[0], epinfos, *map(sf0, (off_policy_ratio, off_env_ratio, ratio)))


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(1, 2).reshape(s[0], s[1] * s[2], *s[3:])


def sf0(arr):
    """
    swap and flatten axis 0 and 1
    """
    return arr.swapaxes(0, 1).ravel()
