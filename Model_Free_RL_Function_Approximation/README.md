# Model Free Reinforcement Learning using Function Approximation

This folder, named **Model_Free_RL_Function_Approximation** implements Deep Q-Network (DQN), which approximates Q-function 
as a Neural Networks. Here, the classic RL benchmark named **Cart-Pole** is used as the target MDP environment. Specific role of each script is listed as follows:


***

## Deep Q-Network (DQN)

DQN learns optimized policies that maximize total rewards using Q-function approximated by Neural Networks. 
For the successful learning, three tips are used as follows:

- Experience Replay
- Fixed Target Q-Network
- Reward Clipping

You can see the effects of these tips against learning optimized policies by changing its learning conditions. 
The main scripts have a dictionary named **hyperparams_CartPole**. By changing its parameters, you can check DQNs'
learning performance as following conditions.

- DQN w/o a replay buffer and w/o a target network.
    - memory_size=1, update_steps=1, batch_size=1, use_target_model=False
- DQN w/o a replay buffer and w/ a target network.
    - memory_size=1, update_steps=1, batch_size=1, use_target_model=True
- DQN w/ a replay buffer and w/o a target network.
    - memory_size=2000, update_steps=10, batch_size=32, use_target_model=False
- DQU w/ a replya buffer and w/ a target network. (Full DQN implementation)
    - memory_size=2000, update_steps=10, batch_size=32, use_target_model=True

***

## Main Scripts

#### main_non_distributed_DQN.py

You can test DQN algorithm w/o distributed computation by running this script. 

#### main_distributed_DQN.py

You can test DQN algorithm w/ distributed computation by running this script.

***

## Role of Other Scripts

#### custom_cartpole.py

This script defines the MDP environment. As mentioned in the above, the environment is the RL benchmark, Cart-Pole. 

#### plot_result.py

This script defines methods to plot a result of reward history.
