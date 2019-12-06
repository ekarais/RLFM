# RLFM
Training RL-based agents in the double-auction market setting.

> * Reinforced BEYBs
> * Group participants names: Ege Karaismailoglu, Burak Alp Kaya, Batuhan Tomekce, Shengyang Zhou
> * Project Title: Reinforcement learning in double auction markets

## General Introduction

Double auction markets have existed throughout history and have been used as a tool to study the convergence of prices in ordinary markets. For us, they are attractive to study because it is relatively easier to reason about optimal strategies in double auction markets, especially if the participants follow a narrow set of policies. 
In addition to their usefulness for economic theories, double auction markets also form a good environment to develop and compare reinforcement learning algorithms. 
Using reinforcement learning, we can simulate the convergence of prices, and given the agents model the reality well enough, arrive at conclusions about real world markets. 

## The Model

Hence, the combination of double auction markets and reinforcement learning is attractive for two reasons. First, reinforcement learning allows us to train agents which display intelligent behavior, i.e., the kind of behavior we would expect from strategically thinking humans. This allows us to explain certain  properties of double auction markets, such as the optimal strategy that should be followed. Secondly, we use the double auction market as a framework for developing agents which are guided by reinforcement learning algorithms. The rest of this paper serves both of these goals.

We measure the fitness of our models by their success in the market environment, quantized through the rewards they achieve.

## Fundamental Questions

1) How do different RL algorithms compare?

2) Can RL agents be trained to recognize weaknesses of heuristic agents?

3) Can Multi-Agent settings be trained to converge to identifiable behaviour patterns?

4) Can a group of Agents learn to cooperate by price-fixing?

## Expected Results

1) Different algorithms could perform very differently from eachother

2) Yes.

3) Yes.

4) Yes.

## References 

See Report.

## Research Methods

* Reinforcement Learning
    * Q Learning
    * Monte Carlo Control
    * DQN
