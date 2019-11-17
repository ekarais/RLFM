#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import warnings

import agents
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
num_episodes = 300
num_sellers = 5 #not used for now
num_buyers = 5 #not used for now
max_budget = 100
init_offer = 0
min_ask_price = 20
init_ask_price = 100


# Creating data structures
names = ['Bob', 'Alice', 'Eve', 'John', 'Nick', 'Giannis', 'Joel', 'Ben', 'Furkan', 'Kevin']

# Sellers
sellers_dict = dict.fromkeys(names[0:5])
sellers = []
for i in range(0, 5):
    s = agents.Seller('Seller ' + names[i], min_ask_price)
    sellers.append(s)
    sellers_dict[names[i]] = s
    
# Buyers
buyers_dict = dict.fromkeys(names[5:10])
buyers = []
for i in range(5, 10):
    b = agents.Buyer('Buyer ' + names[i], max_budget)
    buyers.append(b)
    buyers_dict[names[i]] = b
    
# To store rewards of all agents
aux_1 = []
for s in sellers:
    aux_1.append(s.agent_id)
for b in buyers:
    aux_1.append(b.agent_id)
    
total_rewards = dict.fromkeys(aux_1, 0)

#
Q = np.random.rand(11,11)
C = np.zeros((11,11))
gamma = 0.9
# may have to create the greedy policy w.r.t. Q here, to make bheavior policy epsilon-soft.

# Creating the market
market = MarketEnvironment(sellers=sellers, buyers=buyers, max_steps=10,  
    matcher=matchers.RandomMatcher(reward_on_reference=True), setting=info_settings.BlackBoxSetting)


for i in range (0, num_episodes):
    init_observation = market.reset()

    step1_offers = {}
    for seller in sellers:
        step1_offers[seller.agent_id] = init_ask_price

    for buyer in buyers:
        step1_offers[buyer.agent_id] = init_offer
        buyer.actions.append(init_offer)

    #First step in the market
    observations, rewards, done, _ = market.step(step1_offers)

    for k in total_rewards.keys():
        total_rewards[k] += rewards[k]

        if k.split()[1] in names[0:5]:
            sellers_dict[k.split()[1]].rewards.append(rewards[k])
        else:
            buyers_dict[k.split()[1]].rewards.append(rewards[k])

    # Rest:
    while market.time < market.max_steps:
        offers = {}
        for s in sellers:
            state = get_state(s.agent_id, market)
            offers[s.agent_id] = s.get_new_offer(state)

        for b in buyers:
            state = get_state(b.agent_id, market)
            new_offer = b.get_random_offer(state)
            offers[b.agent_id] = new_offer
            if(market.done[b.agent_id] == False):
                b.actions.append(new_offer)

        observations, rewards, done, _ = market.step(offers)

        for k in total_rewards.keys():
            total_rewards[k] += rewards[k]
            
            key = k.split()[1]

            if key in names[0:5] and sellers_dict[key].done == False:
                sellers_dict[key].rewards.append(rewards[k])
                sellers_dict[key].done = market.done[k]

            if key in names[5:10] and buyers_dict[key].done == False:
                buyers_dict[key].rewards.append(rewards[k])
                buyers_dict[key].done = market.done[k]

    # Learning the q function
    for b in buyers:
        G = 0
        W = 1
        for i in range(0, len(b.rewards)-2):
            t = len(b.rewards) - 2 - i
            S_t = b.actions[t]//10
            A_t = b.actions[t+1]//10

            G = gamma*G + b.rewards[t+1]
            C[S_t, A_t] = C[S_t, A_t] + W
            Q[S_t, A_t] = Q[S_t, A_t] + (W/C[S_t, A_t])*(G - Q[S_t, A_t])
            if A_t != policy(S_t, Q):
                continue
            W = W * (1/behavior_random_p(A_t, S_t))

print('')        
print(pd.DataFrame(market.deal_history))
print('')        
print(market.offers)
print('')        
print(pd.DataFrame(total_rewards, index=[0]))
print('')        

with np.printoptions(precision=2, suppress=True):
    print(Q)

for i in range(0, 11):
    print(f"Last offer = {i*10}  ==> Next offer = {policy(i, Q)}")

np.save('Q_values.npy', Q)
