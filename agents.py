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
        
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        return '(' + self.agent_id + ', ' + str(self.reservation_price) + ')'
    
    def get_new_offer(self, last_offer):
        '''
        Default action function for non-learning agents in the blackbox setting. The next bid is a random variable drawn 
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
        Default action function for non-learning agents in the blackbox setting. The next bid is a random variable drawn 
        between the last bid and budget. If the last bid was too close to the budget, the next bid is the whole budget. 
        '''
       
        if self.agent_id.split()[0] == 'Seller':
            return np.random.random_integers(low=2, high=10) * 10
        else:
            return np.random.random_integers(low=0, high=10) * 10

        
class Buyer(MarketAgent):    
    
    pass


class Seller(MarketAgent):
    
    pass
       
        
    
    
'''
Old, unnecessary initialization code for buyer and seller:

def __init__(self, agent_id: str,  reservation_price: float):
        """
        A buyer agent that extends the market agent
        :param agent_id: a unique id that differentiates this agent to other agents
        :param reservation_price: the reservation price, or maximum price that this agent is
        willing to buy
        """
        super().__init__(agent_id, reservation_price)
        
'''