# Reinforment Learning: Policy-based Algorithms



### Markov Chain

A Markov Chain has a set of status $S = \{s_0, s_1, ..., s_m \}$, and a process that can move successively from one state to another. Each move is a single **step** and is based on a **transition model** $T$. A Markov Chain is based on the **Markov Property**, which states **given the present, the future is conditionally independent of the past**. That is, the state in which the process is now it is dependent only from the state it was at t-1

1. States: $S = \{s_0, s_1, ..., s_m \}$
2. Initial state: $s_0$
3. Transition model: $T(s, s')$

E.g.

> Let’s suppose we have a chain with only two states $s_0$ and $s_1$, where $s_0$ is the initial state. The process is in $s_0$ 90% of the time and it can move to $s_1$ the remaining 10% of the time. When the process is in state $s_1$ it will remain there 50% of the time.

Graphically
$$
\begin{align}
P(s_0|s_0) & = 0.9 \\
P(s_1|s_0) & = 0.1 \\
P(s_0|s_1) & = 0.5 \\
P(s_1|s_1) & = 0.5 \\
\end{align}
$$
And the transition matrix T is 
$$
T = \begin{bmatrix}
0.90 &  0.10 \\
0.50 & 0.50\\
\end{bmatrix}
$$
The basic assuptions are **chains having finite status and transition matrix not changed (time-homogeneous)**. Then we can compute the k-step transition probability as the k-th power of the trainsition matrix by calling `np.linalg.matrix_power()`

**Equilibrium**

The final state is `np.dot(init, trans_matrix)`, and the chain converge to equilibrium meaning that as the time progresses it **forgets about the starting distribution**, while **the convergence is not always guaranteed**. The dynamics of a Markov chain can be very complex, in particular it is possible to have **transient** and **recurrent** states

### Markov Decision Process

A MDP is a reinterpretation of Markov chains which includes an **agent** and a **decision making** process. A MDP is defined by these components

1.  Set of possible States: $S = \{s_0, s_1, ..., s_m \}$
2.  Initial State: $s_0$
3.  Set of possible Actions: $A = \{a_0, a_1, ..., a_m \}$
4.  Transition Model: $T(s, a, s')$
5.  Reward Function: $R(s)$

### Reference 

[Dissecting Reinforcement Learning-Part.1](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html)

[Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)

