# Reinforment Learning: Policy-based Algorithms

### Intro

There are two basic types of approaches allowing agents to learn good behavior, policy gradient and value functions.

#### Policy Gradient

The neural network learns a policy for picking actions by adjusting it's weights through gradient descent using feedback from the environment.

#### Value function

The agent learns to predict how good a given state or action will be for the agent to be in.

### Four-Armed Bandit Example

```python
bandits = [0.2,0,-0.2,-5]
num_bandits = len(bandits)
def pullBandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1
    
tf.reset_default_graph()

weights = tf.Variable(tf.ones([num_bandits]))  # This initialization is one of best
chosen_action = tf.argmax(weights,0)

reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)  # Current reward
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)  # Current action
responsible_weight = tf.slice(weights,action_holder,[1])  # The weights of current action
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

total_episodes = 1000
total_reward = np.zeros(num_bandits)
e = 0.1

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
        reward = pullBandit(bandits[action])
        
        _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
        
        total_reward[action] += reward
        if i % 50 == 0:
            print "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward)
        i+=1
        
print "The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising...."
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print "...and it was right!"
else:
    print "...and it was wrong!"
```

The network would consists of a set of weights, and would represent how good our agent thinks it is to pull each arm. If we initialize these weights to 1, then our agent will be somewhat optimistic about each arm's potential reward.

To update our network, we will simply try an arm with an **e-greedy** policy. This means that **most of the time our agent will choose the action that corresponds to the largest expected value, but occasionally, with e probability, it will choose randomly**. In this way, the agent can **try out each of the different arms to continue to learn more about them**. Once our agent has taken an action, it then receives a reward of either 1 or -1. With this reward, we can then make an **update to our network using the policy loss equation**
$$
Loss = Log(\pi) * A
$$
$A$ is advantage, and is an essential aspect of all reinforcement learning algorithms. Intuitively it corresponds to how much better an action was than some baseline. In this case, we assume the baseline is 0, hence advantage is the reward we recieved for each action.

$\pi$ is the policy. In this case, it corresponds to the chosen action's weight.

Intuitively, this loss function allows us to increase the weight for actions that yielded a positive reward, and decrease them for actions that yielded a negative word.

### Contextual Bandit

![](/assets/contextual_bandits.png)

Contextual Bandits introduce the concept of the **state**. **The state consists of a description of the environment that the agent can use to take more informed actions**. The objective function of this problem is to maximize many bandits, whereas the former case is to maximize a single bandits.

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
5.  Reward Function: $R(s, a)$

As such, at any time in an MDP, an agent is given a state $s$, takes action $a$, and receives new state $s'$ and reward $r$, and the main distinction of MDP with Markov Chain is that the **transition model depends on the current state, the next state and the action of agent**. The transition model returns the probability of reaching the state $s'$ if the action $\alpha$ is done in state $s$. But given $s$ and $a$ the model is conditionally independent of all previous states and actions

Hence the problem changes to the agent has to **maximise the reward avoiding states which return negative values and choosing the one which return positive value**. The solution is to **find a policy $\pi(s)$ which returns the action with the highest reward**

### The Bellman Equation

The bellman equation is the solution of **how the agent choose the best policy**. First of all we have to find a way to compare two policies.
$$
U(s) = E[\sum^{\infty}_{t=0}\gamma^tR(s_t)] = R(s_0) +\gamma R(s_1) + ... + \gamma^nR(s_n)
$$
Then, the utility of a state $s$ is correlated with the utility of its neighbors at $s'$
$$
U(s) = R(s) +\gamma max_{a}\sum_{s'}T(s, a, s')U(s')
$$
E.g.



### Cart-Pole

> A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
>
> CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.

This task requires:

* observations: The agent needs to know where pole currently is, and the angle at which it is balancing. To accomplish this, our neural network will take an observation and use it when producing the probability of an action
* Delayed reward: Keeping the pole in the air as long as possible means moving in ways that will be advantageous for both the present and the future. To accomplish this we will adjust the reward value for each observation-action pair using a function that weighs actions over time

### Reference 

[Dissecting Reinforcement Learning-Part.1](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html)

[Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)

[Simple Reinforcement Learning in Tensorflow: Part 1 - Two-armed Bandit](https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149)

[Simple Reinforcement Learning with Tensorflow Part 1.5: Contextual Bandits](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c)

[Simple Reinforcement Learning with Tensorflow: Part 2 - Policy-based Agents](https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724)

[CartPole-v0](https://gym.openai.com/envs/CartPole-v0)

