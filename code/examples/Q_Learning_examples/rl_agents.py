from agents import Seller, Buyer
import numpy as np 
import random

class RLSeller(Seller):
    def __init__(self, agent_id: str, reservation_price: float):

        super().__init__(agent_id, reservation_price)

    # get q_table as input to change, epsilon_decision is for explore or test
    def update_offer(self, next_step, state, q_table = None, epsilon_decision = True):

        '''

        next_step ... is to update the actions dictionary to be able to play one step of the game
        state ... get your last action from the environment
        q_table ... q_table to get the actions
        epsilon_decision ... decides to choose random actions or depending on the q_table


        updates the actions dictionary from the environment to play one step of the game
        returns the previous offer of the agent and the chosen action from q table

        '''
    

        # get the agents last offer from the state dictionary.
        last_offer = state[0]

        # to choose a random action
        random_action = random.randint(0,2)

        # if at the beginning hard code the default starting value for offering
        if last_offer == 0:
            offer = {self.agent_id: self.reservation_price * 2}
            chosen_action = 2

        # else decide what to do
        else:
            # to get the index in q table
            index = (last_offer - self.reservation_price) / 5

            # to choose if we are going to choose random actions or if we are going to use the q table.
            if epsilon_decision:
                chosen_action = random_action
                 # Explore action space

            # get the action from q table and calculate what to offer next
            # use integer conversion since offers are float
            else:
                chosen_action = np.argmax(q_table[int(index)]) # Exploit learned values
            if chosen_action == 0:
                offer_value = last_offer - 25
            elif chosen_action == 1:
                offer_value = last_offer - 5
            else:
                offer_value = last_offer

            # if the offer is below 100 than always offer 100 to not lose money
            if offer_value < 100:
                offer_value = 100

            # create dictionary element
            offer = {self.agent_id: offer_value}
        
        # update the dictionary add the new element
        next_step.update(offer)

        # return the previous offer and chosen action in order to update the q values.
        return last_offer, chosen_action
