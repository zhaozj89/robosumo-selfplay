import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.reset()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path+'_Q.npy', self.Q)
        utils.save(model_path+'_N.npy', self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path+'_Q.npy')
        self.N = utils.load(model_path+'_N.npy')

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead, opponent="False"):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately
        '''
        state_old = state
        state = []
        if opponent==False:
            state.append(state_old[0])
            state.append(state_old[1])
            state.append(state_old[4])
            state.append(state_old[6])
            state.append(state_old[7])
        else:
            state.append(state_old[2])
            state.append(state_old[3])
            state.append(state_old[5])
            state.append(state_old[6])
            state.append(state_old[7])
        if self.a == None:
            self.s = self.discretize(state)
            self.a = self.actions[-1]
            return self.a
        R = -0.1
        if points > self.points:
            R = 1
            self.points += 1
        if (self.s[4] == 1 and self.s[5] == 1 and self.s[6] == 1) or (self.s[5] == 1 and self.s[6] == 1 and self.s[7] == 1) or (self.s[4] == 1 and self.s[6] == 1 and self.s[7] == 1) or (self.s[4] == 1 and self.s[5] == 1 and self.s[7] == 1):
            R = -0.5

        s_prime = self.discretize(state)
        if self._train:
            self.N[self.s][self.a] += 1
        alpha = self.C/(self.C+self.N[self.s][self.a])
        
        if dead:
            R = -1
            self.Q[self.s][self.a] = self.Q[self.s][self.a] + alpha*(R + self.gamma*np.max(self.Q[s_prime][self.actions]) - self.Q[self.s][self.a])
            self.reset()
        else:
            self.Q[self.s][self.a] = self.Q[self.s][self.a] + alpha*(R + self.gamma*np.max(self.Q[s_prime][self.actions]) - self.Q[self.s][self.a])
            self.s = s_prime
            self.a = np.max(np.argwhere(self.f(self.Q[self.s],self.N[self.s]) == np.amax(self.f(self.Q[self.s],self.N[self.s]))))
            if not self._train:
                self.a = np.max(np.argwhere(self.Q[self.s] == np.amax(self.Q[self.s])))
            
        return self.a

    def f(self,u,n):
        b1 = n < self.Ne
        b2 = np.invert(b1)
        return b1 + b2*u

    def discretize(self, state):
        adjoining_wall_x = 0
        if state[0]//40 == 1:
            adjoining_wall_x = 1
        if state[0]//40 == 10:
            adjoining_wall_x = 2
        
        adjoining_wall_y = 0
        if state[1]//40 == 1:
            adjoining_wall_y = 1
        if state[1]//40 == 10:
            adjoining_wall_y = 2
        
        food_dir_x = 0
        if state[3] < state[0]:
            food_dir_x = 1
        if state[3] > state[0]:
            food_dir_x = 2
        
        food_dir_y = 0
        if state[4] < state[1]:
            food_dir_y = 1
        if state[4] > state[1]:
            food_dir_y = 2

        snake_body = np.floor_divide(state[2],40)
        adjoining_body_left = 0
        adjoining_body_right = 0
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        for seg in snake_body:
            if np.array_equal([state[0]//40 + 1,state[1]//40], seg):
                adjoining_body_left = 1
            if np.array_equal([state[0]//40 - 1,state[1]//40], seg):
                adjoining_body_right = 1
            if np.array_equal([state[0]//40,state[1]//40 + 1], seg):
                adjoining_body_top = 1
            if np.array_equal([state[0]//40,state[1]//40 - 1], seg):
                adjoining_body_bottom = 1
            if np.array_equal([state[0]//40 + 1,state[1]//40 + 1], seg) and np.array_equal([state[0]//40 + 1,state[1]//40 - 1], seg) and np.array_equal([state[0]//40 - 1,state[1]//40], seg):
                adjoining_body_right = 1
                adjoining_body_bottom = 1
                adjoining_body_top = 1
            if np.array_equal([state[0]//40 - 1,state[1]//40 + 1], seg) and np.array_equal([state[0]//40 + 1,state[1]//40 + 1], seg) and np.array_equal([state[0]//40,state[1]//40 - 1], seg):
                adjoining_body_bottom = 1
                adjoining_body_left = 1
                adjoining_body_right = 1
            if np.array_equal([state[0]//40 - 1,state[1]//40 + 1], seg) and np.array_equal([state[0]//40 - 1,state[1]//40 - 1], seg) and np.array_equal([state[0]//40 + 1,state[1]//40], seg):
                adjoining_body_left = 1
                adjoining_body_top = 1
                adjoining_body_bottom = 1
            if np.array_equal([state[0]//40 - 1,state[1]//40 - 1], seg) and np.array_equal([state[0]//40 + 1,state[1]//40 - 1], seg) and np.array_equal([state[0]//40,state[1]//40 + 1], seg):
                adjoining_body_top = 1
                adjoining_body_left = 1
                adjoining_body_right = 1
        
        return (adjoining_wall_x,adjoining_wall_y,food_dir_x,food_dir_y,adjoining_body_top,adjoining_body_bottom,adjoining_body_left,adjoining_body_right)