__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

import numpy as np
import random

class MarketAgent:
    def __init__(self, agent_id: str, reservation_price: float):
        """
        An market agent object. This class is extended to include all the agent logic for the
        agent interactions.
        :param agent_id: A unique id to differentiate this agent with other agents
        :param reservation_price: the reservation price, which is agents ideal price regarding a
        purchase or sale
        """
        self.agent_id = agent_id
        self.reservation_price = reservation_price
        self.actions = [] # to enable off-policy MC
        self.rewards = []
        self.done = False
        self.total_rewards = 0 # to enable comparing agents' performances
        self.eval = False # Flag determining if the agent is in evaluation mode (to stop learning)

    def __repr__(self):
        return str(self)
    
    def __str__(self):
        return '(' + self.agent_id + ', ' + str(self.reservation_price) + ')'
    
    def evaluate(self):
        '''
        Puts the agent in evaluation mode. For non-learning agents, there should be no difference
        between eval mode switched on and off.
        '''
        self.eval = True

    def get_new_offer(self, last_offer):
        '''
        Generic action function for non-learning agents in the blackbox setting. The next bid is a random variable drawn 
        between the last bid and budget. If the last bid was too close to the budget, the next bid is the whole budget. 
        '''
        if last_offer == self.reservation_price:
            return self.reservation_price
        
        else:
            if last_offer <= self.reservation_price: #this is a buyer
                a = last_offer
                b = self.reservation_price
            else: #this is a seller
                a = self.reservation_price
                b = last_offer
            return np.random.random_integers(low=a//10, high=b//10) * 10

    def get_random_offer(self, last_offer):
        '''
        Generic action function for non-learning agents in the blackbox setting. The next bid is a random variable drawn 
        uniformly from the set of all possible actions. 
        '''
       
        if (isinstance(self, Seller)):
            return np.random.random_integers(low=2, high=10) * 10
        else:
            return np.random.random_integers(low=0, high=10) * 10


    def get_random_offer_p(self, A, S):
        '''
        The probabililty distribution for the policy implemented by get_random_offer():
        '''
        return 1/11

    def set_done(self):
        '''
        Sets the agent's done field. For agents which learn at the end of every episode, this
        function should implement the learning algorithm. See MonteCarlo_MarketAgent for an example.
        '''

        self.done = True


    def reset(self):
        self.actions = [] 
        self.rewards = []
        self.done = False

class Buyer(MarketAgent):    
    
    pass


class Seller(MarketAgent):
    
    pass
       
        
class MonteCarlo_MarketAgent(MarketAgent):

    def __init__(self, agent_id: str, reservation_price: float, b_policy: callable, b_policy_dist: callable):
        super().__init__(agent_id, reservation_price)
        # assumes the state-action space is fixed (0 to 100, 11 states, 11 actions)
        self.Q_table = np.random.rand(11,11)
        self.C_table = np.zeros((11,11))
        self.gamma = 1
        self.b_policy = b_policy
        self.b_policy_dist = b_policy_dist
        self.behavior_buyer = Buyer('Behavior', self.reservation_price) # needed because some 
                                                        # MarketAgent functions require 'self'
        

    def set_done(self):
        '''
        Sets the agents done field, and performs an iteration of the Monte Carlo algorithm.
        '''
        self.done = True

        if (self.eval == True):
            return

        G = 0
        W = 1
        for i in range(0, len(self.rewards)-2):
            t = len(self.rewards) - 2 - i
            S_t = self.actions[t]//10
            A_t = self.actions[t+1]//10

            G = self.gamma*G + self.rewards[t+1]
            self.C_table[S_t, A_t] = self.C_table[S_t, A_t] + W
            self.Q_table[S_t, A_t] = self.Q_table[S_t, A_t] + (W/self.C_table[S_t, A_t])*(G - self.Q_table[S_t, A_t])
            
            if (A_t != self.target_policy(S_t)):
                continue

            W = W * (1/self.b_policy_dist(self.behavior_buyer, A_t, S_t))

    def target_policy(self, S_t):
        '''
        Given a state, returns the action according to the agent's computed target policy.
        '''

        return np.argmax(self.Q_table[S_t//10])*10

    
