# Synchronous Policy Optimization w/o and w/ Distributed Computing

This folder, named **MDP** implements two types of policy optimization methods for MDP environments: synchronous value iteration (VI) and policy iteration (PI).
Here, the grid-world called **FrozenLake** is regarded as the target MDP environment. Specific role of each script is listed as follows:

***

## Synchronous Value Iteration

The first algorithm for the policy optimization in the given MDP is **synchronous value iteration (VI)** based on Bellman backups. 
The optimized policy is obtained as a greedy policy. Following three files are main algorithms of synchronous VI and for executing them at one time.

#### sync_value_iteration.py

This script implements the synchronous non-distributed VI. The method takes environment, discount factor, and threshold to stop the iteration and returns optimized state values with greedy policy.

#### sync_distributed_value_iteration.py

This script implements the synchronous distributed VI. Detailed description will be provided in future.

#### main_VI.py

You can test synchronous VI w/o and w/ distributed computing by running this script. As optional, there are several sizes of MDP environments.

***

## Synchronous Policy Iteration

The second algorithm for the policy optimization in the given MDP is **synchronous policy iteration (PI)**. First, a random policy is set as the initial policy.
The policy is evaluated with restricted Bellman backups then improved using Q-function. Following three files are main algorithms of synchronous VI and for executing them at one time.

#### sync_policy_iteration.py

This script implements the synchronous non-distributed PI. The method takes environment, discount factor, and threshold to stop the iteration and returns optimized state values with greedy policy.

#### sync_distributed_policy_iteration.py

Under implementation...

#### main_PI.py

You can test synchronous PI w/o and w/ distributed computing by running this script. As optional, there are several sizes of MDP environments.

***

## Role of Other Scripts

#### environment.py

This script defines the MDP environment. As mentioned in the above, the environment is expressed as grid-world. 

#### policy_evaluation.py

This script defines two methods to evaluate optimized policy after finishing VI/PI.

#### summarize_results.py

This script defines a method to summarize results such as final state value and policy, as well as average discounted value and image view as a heatmap.
