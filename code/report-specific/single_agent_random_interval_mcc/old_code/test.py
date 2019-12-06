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

import sys
sys.path.append("../..")
sys.path.append("../../old_code")

from plot_config import *

import info_settings
import matchers

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

# Utility functions
def get_state(agent_id, market):
    '''
    A cleaner way of getting the state for a user. Assumes that information setting is black-box.
    '''
    return market.setting.get_state(agent_id, market.deal_history, market.agents, market.offers)[0]
    

#Problem settings
num_episodes = 1
num_sellers = 5 
num_buyers = 5 
max_budget = 100
init_offer = 0
min_ask_price = 20
init_ask_price = 100
non_rl_policy = agents.MarketAgent.get_new_offer

## name of the RL agent
rl_agent = "MCC"

# Creating data structures
names = ['H1_s', 'H2_s', 'H3_s', 'H4_s', 'H5_s', 'H6_b', 'H7_b', 'H8_b', 'H9_b', rl_agent]

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
        Q = np.load('Q_values.npy')
        b.Q_table = Q
        b.evaluate() # sets b to evaluating mode, to avoid updating the Q table

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
                new_offer = a.target_policy(state) # call your RL algorithm here
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
sum = 0
for a_id, a in agent_dict.items():
    total_rewards[a_id] = a.total_rewards
    
for a_id, a in total_rewards.items():
    total_rewards[a_id] /= total_rewards[rl_agent]
    
print(pd.DataFrame(total_rewards, index=[0]))

ranking = sorted(total_rewards.items(), key = 
             lambda kv:(kv[1], kv[0]))

print(ranking[0])

ranking.reverse()
scores = []
namess = []
for i in range (0, 10):
    scores.append(ranking[i][1])
    namess.append(ranking[i][0])
print(type(ranking))



objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(10)
performance = [10,8,6,4,2,1]

plt.bar(y_pos, scores[0:10], align='center', alpha=0.5)
plt.xticks(y_pos, namess)
plt.ylabel('Normalized rewards ')
plt.title('Average performance of agents')

#fig = plt.figure()
#fig.set_size_inches(w=4.7747, h=4.)
plt.savefig('single_agent_random_interval_mcc.pgf')
