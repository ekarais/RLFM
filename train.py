#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import warnings
import agents
from agents import Seller, Buyer
from environments import MarketEnvironment


import info_settings
import matchers

# Utility functions
def get_state(agent_id, market):
    '''
    A cleaner way of getting the state for a user. Assumes that information setting is black-box.
    '''
    return market.setting.get_state(agent_id, market.deal_history, market.agents, market.offers)[0]
    
def policy(S_t, Q):
    return np.argmax(Q[S_t])*10

def behavior_p(A, S):
    return 1/((100 - S)//10 + 1)

def behavior_random_p(A, S):
    return 1/11

#Problem settings
num_episodes = 100
num_sellers = 4 
num_buyers = 5 
max_budget = 100
init_offer = 0
min_ask_price = 20
init_ask_price = 100
non_rl_policy = agents.MarketAgent.get_random_offer
## name of the RL agent
rl_agent = "Jarvis"

# Creating data structures
names = ['Alice', 'Eve', 'John', 'Nick', 'Giannis', 'Joel', 'Ben', 'Furkan', rl_agent]

agent_list = [] # this exists because the MarketEnvironment wants lists
agent_dict = {} # the code will use this because it is much more handy

for i in range(0, num_sellers):
    s = Seller(names[i], min_ask_price)
    agent_list.append(s)
    agent_dict[names[i]] = s

for i in range(num_sellers, num_sellers + num_buyers):
    if (names[i] != rl_agent):
        b = Buyer(names[i], max_budget)

    # initialize your agent here (if it is a buyer)    
    else:
        b = agents.MonteCarlo_MarketAgent(names[i], max_budget, agents.MarketAgent.get_random_offer, agents.MarketAgent.get_random_offer_p)

    agent_list.append(b)
    agent_dict[names[i]] = b


# Creating the market
market = MarketEnvironment(sellers=agent_list[0:num_sellers], buyers=agent_list[num_sellers:num_sellers+num_buyers], max_steps=10,  
    matcher=matchers.RandomMatcher(reward_on_reference=True), setting=info_settings.BlackBoxSetting)


for i in range (0, num_episodes):
    init_observation = market.reset()
    
    # resetting the agents
    for a_id, a in agent_dict.items():
        a.reset()

    step1_offers = {}
    for a_id, a in agent_dict.items():
        # This if-clause will be problematic in the future. Right now, any RL seller won't be
        # recognized as a seller. To solve, one can create a list of seller classes, and see
        # if the instance's class belongs to the list.
        if (isinstance(a, Seller)):
            step1_offers[a_id] = init_ask_price
            a.actions.append(init_ask_price)
        else:
            step1_offers[a_id] = init_offer
            a.actions.append(init_offer)
    
    #First step in the market
    observations, rewards, done, _ = market.step(step1_offers)

    for k in rewards.keys():
        agent_dict[k].total_rewards += rewards[k]
        agent_dict[k].rewards.append(rewards[k])

    # Rest:
    while market.time < market.max_steps:
        offers = {}

        for a_id, a in agent_dict.items():
            state = get_state(a_id, market)

            if (a_id == rl_agent):
                new_offer = a.b_policy(a.behavior_buyer, state) # call your RL algorithm here
            else:
                new_offer = non_rl_policy(a, state)

            offers[a_id] = new_offer

            if (market.done[a_id] == False):
                a.actions.append(new_offer)

        
        observations, rewards, done, _ = market.step(offers)

        for k in rewards.keys():
            if (agent_dict[k].done == False):
                agent_dict[k].total_rewards += rewards[k]
                agent_dict[k].rewards.append(rewards[k])
                # Instead of setting the done field manually, call the done function of the agent, 
                # because some RL algorithms train not after every step, but after every episode.
                # Such algorithms can silently run by calling their done() function.
                if (market.done[k]):
                    agent_dict[k].set_done()
                

    

total_rewards = {}
for a_id, a in agent_dict.items():
    total_rewards[a_id] = a.total_rewards

print(pd.DataFrame(total_rewards, index=[0]))
print('')        

with np.printoptions(precision=2, suppress=True):
    print(agent_dict[rl_agent].Q_table)

for i in range(0, 11):
    print(f"Last offer = {i*10}  ==> Next offer = {agent_dict[rl_agent].target_policy(i*10)}")

np.save('Q_values.npy', agent_dict[rl_agent].Q_table)
