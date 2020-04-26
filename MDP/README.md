### Synchronous Value Iteration w/o and w/ Distributed Computing

This folder, named **MDP** implements synchronous value iteration w/o and w/ distributed computing. Here, the grid-world called **FrozenLake** 
is regarded as the target MDP environment. The state value is iteratively updated until the Bellman error will be below a specified threshold.
Specific role of each script is listed as follows:

#### environment.py

This script defines the MDP environment. As mentioned in the above, the environment is expressed as grid-world. 

#### sync_value_iteration.py

This script implements the synchronous non-distributed VI. The method takes environment, discount factor, and threshold to stop the iteration and returns optimized state values with greedy policy.

#### sync_distributed_iteration.py

This script implements the synchronous distributed VI. Detailed description will be provided in future.

#### main_VI.py

You can test synchronous VI w/o and w/ distributed computing by running this script. As optional, there are several sizes of MDP environments.