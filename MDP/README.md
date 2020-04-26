### Synchronous Value Iteration w/o and w/ Distributed Computing

This folder, named **MDP** implements synchronous value iteration w/o and w/ distributed computing. Here, the grid-world called **FrozenLake** 
is regarded as the target MDP environment. The state value is iteratively updated until the Bellman error will be below a specified threshold.
Specific role of each script is listed as follows:

#### environment.py

This script defines the MDP environment. As mentioned in the above, the environment is expressed as grid-world. 

#### sync_value_iteration.py

This script implements the synchronous non-distributed VI. The method takes environment, discount factor, and threshold to stop the iteration and returns optimized state values with greedy policy.

#### sync_distributed_iteration.py

This script implements the synchronous distributed VI. 

![sequence dialog](http://www.plantuml.com/plantuml/proxy?src=https://gist.githubusercontent.com/masafumiendo/a4066e10514c4cea564a7b9691f994ca/raw)
https://gist.github.com/masafumiendo/a4066e10514c4cea564a7b9691f994ca

<div class="flow">
st=>start: Start:>http://www.google.com[blank]
e=>end:>http://www.google.com
op1=>operation: My Operation
sub1=>subroutine: My Subroutine
cond=>condition: Yes
or No?:>http://www.google.com
io=>inputoutput: catch something...

st->op1->cond
cond(yes)->io->e
cond(no)->sub1(right)->op1
</div>

#### main_VI.py

You can test synchronous VI w/o and w/ distributed computing by running this script. As optional, there are several sizes of MDP environments.