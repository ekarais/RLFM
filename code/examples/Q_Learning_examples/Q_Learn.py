import numpy as np
import random
from dmarket.agents import MarketAgent

class StochasticDescendingAgent(MarketAgent):
    """
    Stocasticaly decreases or increases offer
    """
    
    def __init__(self, role, reservation_price, name=None):
        super().__init__(role, reservation_price, name)
    def get_offer(self,observation):
        last_offer = observation[0]
        offer = self.reservation_price
        if self.role == 'seller':
            if last_offer == 0:
                offer = self.reservation_price * 2
            elif last_offer <= self.reservation_price:
                offer = self.reservation_price
            else:
                a = self.reservation_price + (last_offer - self.reservation_price)* (2/3)
                offer_value = random.uniform(a,last_offer)
        elif self.role == 'buyer':
            if last_offer == 0:
                offer = self.reservation_price / 2
            elif last_offer >= self.reservation_price:
                offer = self.reservation_price
            else:
                offer_value = random.uniform(0,last_offer*(1/5))
                offer = offer_value + last_offer
        return offer
    
class SpecialRandomAgent(MarketAgent):
    def __init__(self, role, reservation_price, name=None):
        super().__init__(role, reservation_price, name)
    def get_offer(self,observation):
        if self.role == 'buyer':
            return random.uniform(100,180)
        elif self.role == 'seller':
            return random.uniform(120,200)

    
class QAgent(MarketAgent):
    """
    Performs Q learning in its own table
    """
    def __init__(self, role, reservation_price, name=None, train=True):
        super().__init__(role, reservation_price, name)
        self.train= train
        self.q_table = np.zeros((21,3))
        self.gamma = 0.95
        self.alpha = 0.2
        self.earlier_offer = 0
        self.new_offer = 0
        self.reward = 0
        self.earlier_action = 0
        
        
    def get_offer(self, observation):
        last_offer = observation[0]

        if self.role == 'seller':
            if (last_offer == 0):
                self.earlier_offer = self.reservation_price*2
                self.earlier_action = 2 # staying where you are
                return self.earlier_offer
            else:
                self.earlier_offer = last_offer
                self.new_offer = self.table_to_offer(self.observation_to_table(self.earlier_offer))
                return self.new_offer
    
    # the function that returns the table index from given information
    def observation_to_table(self,offer):
        return int((offer-100)/5)
        
    # function that returns the offer from the given table index
    def table_to_offer(self, table_index):
        if (not self.train):
            self.earlier_action = np.argmax(self.q_table[table_index])
        else:
            self.earlier_action = random.randint(0,2)
        if self.earlier_action == 0:
            offer_value = self.earlier_offer - 25
        elif self.earlier_action == 1:
            offer_value = self.earlier_offer - 5
        else:
            offer_value = self.earlier_offer

            
        # if the offer is below 100 than always offer 100 to not lose money
        if offer_value < 100:
            offer_value = 100
        if offer_value > 200:
            offer_value = 200
        return offer_value
    
    def update_table(self, reward):
        self.reward = reward
        old_value = self.q_table[self.observation_to_table(self.earlier_offer), self.earlier_action]
        next_max = np.max(self.q_table[self.observation_to_table(self.new_offer)])
        new_value = (1 - self.alpha) * old_value + self.alpha * (self.gamma*next_max + self.reward)
        self.q_table[self.observation_to_table(self.earlier_offer), self.earlier_action] = new_value