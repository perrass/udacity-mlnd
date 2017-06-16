# Reinforcement Learning: Q Learning

## Intro

Unlike policy gradient methods, which attempt to learn functions which directly map an observation to an action, Q-learning attempts to learn the value of being in a given state, and taking a specific action there. The toy example for beginner is [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0).

> The FrozenLake environment consists of a 4x4 grid of blocks, each one either being the start block, the goal block, a safe frozen block, or a dangerous hole. The objective is to have an agent learn to navigate from the start to the goal without moving onto a hole. At any given time the agent can choose to move either up, down, left, or right. The catch is that there is a wind which occasionally blows the agent onto a space they didn’t choose. As such, perfect performance every time is impossible, but learning to avoid the holes and reach the goal are certainly still doable. The reward at every step is 0, except for entering the goal, which provides a reward of 1.

## Q-Learning

In it's simplest implemetation, Q-Learning is a table of values for **every state (row)** and **action (column)** possible in the environment. Within each cell of the table, we learn a **value for how good it is to take a given action within a given state**. In the case of FrozenLake environment, the shape of Q-table is $16 \times 4$. We initialize the Q table with zeros or arbitrarily, and then update using [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation), which states that the expected long-term reward for a given action is equal to the immediate reward from the current action combined with the expected reward from the best future action taken at the following state.

> Bellman equation (dynamic programming equation) is a necessary condition for optimally associated with the mathematical optimization method known as dynamic programming

The equation is 
$$
Q(s,a) = r + \gamma(max(Q(s', a')))
$$
This says that the Q-value for a given state (s) and action (a) should represent the current reward (r) plus the **maximum discounted** ($\gamma$) future reward expected according to our own table for the next state ($s'$) we would end up in.

### Algorithms

1. init Q-table arbitrarily
2. observe init state $s_0$
3. repeat until terminated
   1. select and carry out an action *a*
   2. observe reward $r$ and new state $s'$
   3. $Q[s, a] = Q[s, a] + \alpha (r + \gamma max_{a'}(Q[s', a'] - Q[s, a]))$
   4. $s = s'$ 

## Reference

[DEMYSTIFYING DEEP REINFORCEMENT LEARNING](http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/)

[Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)