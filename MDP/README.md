### Synchronous Value Iteration w/o and w/ Distributed Computing

This folder, named **MDP** implements synchronous value iteration w/o and w/ distributed computing. Here, the grid-world called **FrozenLake** 
is regarded as the target MDP environment. The state value is iteratively updated until the Bellman error will be below a specified threshold.
Specific role of each script is listed as follows:

#### environment.py

This script defines the MDP environment. As mentioned in the above, the environment is expressed as grid-world. 

#### sync_distributed_iteration.py
