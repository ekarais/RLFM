from agents import Seller, Buyer
import numpy as np
import random

class AutomaticSeller(Seller):
    def __init__(self,agent_id: str, reservation_price: float):
        
        super().__init__(agent_id, reservation_price)

    def update_offer(self, next_step, state):
        last_offer = state[0]
        if last_offer == 0:
            offer = {self.agent_id: self.reservation_price * 2}
        elif last_offer <= self.reservation_price:
            offer = {self.agent_id: self.reservation_price}
        else:
            a = self.reservation_price + (last_offer - self.reservation_price)* (2/3)
            offer_value = random.uniform(a,last_offer)
            offer = {self.agent_id: offer_value}
        next_step.update(offer)
            

class AutomaticBuyer(Buyer):
    def __init__(self,agent_id: str, reservation_price: float):
        
        super().__init__(agent_id, reservation_price)

    def update_offer(self, next_step, state):
        last_offer = state[0]
        if last_offer == 0:
            offer = {self.agent_id: self.reservation_price / 2}
        elif last_offer >= self.reservation_price:
            offer = {self.agent_id: self.reservation_price}
        else:
            last_offer = state[0]
            
            offer_value = random.uniform(0,last_offer*(1/5))
            offer = {self.agent_id: offer_value + last_offer}
        next_step.update(offer)

class IntervallHalbierungBuyer(Buyer):
    def __init__(self,agent_id: str, reservation_price: float):
        self.initial = True
    
        super().__init__(agent_id, reservation_price)

    def update_offer(self, next_step, state):
        last_offer = state[0]
        if self.initial:
            offer = {self.agent_id: 0}
            self.initial = False
        elif last_offer == self.reservation_price:
            offer = {self.agent_id: self.reservation_price}
        else:
            random_int = np.random.random_integers(last_offer//10,self.reservation_price//10)
            offer = {self.agent_id: random_int * 10}

        next_step.update(offer)

class IntervallHalbierungSeller(Seller):
    def __init__(self,agent_id: str, reservation_price: float):
        self.initial = True
    
        super().__init__(agent_id, reservation_price)

    def update_offer(self, next_step, state):
        last_offer = state[0]
        if self.initial:
            offer = {self.agent_id: 100}
            self.initial = False
        elif last_offer == self.reservation_price:
            offer = {self.agent_id: self.reservation_price}
        else:
            random_int = np.random.random_integers(self.reservation_price//10,last_offer//10)
            offer = {self.agent_id: random_int * 10}

        next_step.update(offer)
        
class SpecialRandomSeller(Seller):
    def __init__(self,agent_id: str, reservation_price: float):
        
        super().__init__(agent_id, reservation_price)

    def update_offer(self, next_step, state):
            
        offer = {self.agent_id: random.uniform(120,200)}
        next_step.update(offer)
            

class SpecialRandomBuyer(Buyer):
    def __init__(self,agent_id: str, reservation_price: float):
        
        super().__init__(agent_id, reservation_price)

    def update_offer(self, next_step, state):

        offer = {self.agent_id: random.uniform(100,180)}
        next_step.update(offer)

        