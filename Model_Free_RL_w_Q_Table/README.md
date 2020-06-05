# Model Free Reinforcement Learning w/ Q-Table

This folder, named **Model_Free_RL_w_Q_Table** implements two types of tabular based model free RL algorithm named 
**SARSA** and **Q-Learning**. Here, the grid-world called **FrozenLake** is regarded as the target MDP environment. Specific role of each script is listed as follows:


***

## Temporal Difference Learning

Temporal difference (TD) is an algorithm that is used for updating the state value function of the MDP. 

#### TD_agent.py

This script implements explore-exploit policy and a method named **learn_and_evaluate** for the TD agent to 
learn and evaluate its policy. 

***

## SARSA

The first algorithm of model-free RL is **SARSA**. SARSA is a on-policy algorithm, which aims to learn
the value of the policy being used to collect data. The following three files are main algorithms of SARSA 
and executing it.

#### SARSA.py

This script implements 

#### main_SARSA.py

You can test synchronous VI w/o and w/ distributed computing by running this script. As optional, there are several sizes of MDP environments.

***

## Q-Learning

The second algorithm for the policy optimization in the given MDP is **synchronous policy iteration (PI)**. First, a random policy is set as the initial policy.
The policy is evaluated with restricted Bellman backups then improved using Q-function. Following three files are main algorithms of synchronous VI and for executing them at one time.

#### Q_Learning.py

This script implements the synchronous non-distributed PI. The method takes environment, discount factor, and threshold to stop the iteration and returns optimized state values with greedy policy.


#### distributed_Q_Learning.py

You can test synchronous PI w/o distributed computing by running this script. As optional, there are several sizes of MDP environments.

#### main_Q_Learning.py

***

## Role of Other Scripts

#### environment.py

This script defines the MDP environment. As mentioned in the above, the environment is expressed as grid-world. 

#### policy_evaluation.py

This script defines two methods to evaluate optimized policy after finishing VI/PI.

#### summarize_results.py

This script defines a method to summarize results such as final state value and policy, as well as average discounted value and image view as a heatmap.
