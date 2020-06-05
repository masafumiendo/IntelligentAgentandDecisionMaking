# Model Free Reinforcement Learning using Q-Table

This folder, named **Model_Free_RL_w_Q_Table** implements two types of tabular based model free RL algorithm named 
**SARSA** and **Q-Learning**. Here, the grid-world called **FrozenLake** is regarded as the target MDP environment. Specific role of each script is listed as follows:


***

## Temporal Difference Learning

Temporal difference (TD) is an algorithm that is used for updating the state value function of the MDP from the agent's experience.

#### TD_agent.py

This script implements explore-exploit policy and a method named **learn_and_evaluate** for the TD agent to 
learn and evaluate its policy. 

***

## SARSA

The first algorithm of model-free RL is **SARSA**. SARSA is a on-policy algorithm, which aims to learn
the value of the policy being used to collect data. The following two files are main algorithms of SARSA 
and executing it.

#### SARSA.py

This script implements the update process of Q-table based on the learned policy. Since it uses the policy, SARSA is called as the on-policy method.

#### main_SARSA.py

You can test SARSA algorithm by running this script. As optional, there are several sizes of MDP environments.

***

## Q-Learning

The second algorith of model-free RL is **Q-Learning**. Unlike SARSA, Q-Learning does not uses the learned policy for the state value estimation.
The following two files are main algorithm of Q-Learning and executing it.

#### Q_Learning.py

This script implements the update process of Q-table based on the greedy policy. Since it does not uses the learned policy, Q-Learning is called 
as the off-policy method.

#### distributed_Q_Learning.py

This script implements distributed version of Q-Learning. Both learning and evaluation processes are conducted in parallel to reduce its running time.

#### main_Q_Learning.py

You can test Q-Learning algorithm by running this script. As optional, there are several sizes of MDP environments.

#### main_distributed_Q_Learning.py

You can test distributed version of Q-Learning algorithm by running this script. As optional, there are several sizes of MDP environments.

***

## Role of Other Scripts

#### environment.py

This script defines the MDP environment. As mentioned in the above, the environment is expressed as grid-world. 

#### summarize_results.py

This script defines methods to summarize results such as plots of reward history and visualized best Q-values with its policy.
