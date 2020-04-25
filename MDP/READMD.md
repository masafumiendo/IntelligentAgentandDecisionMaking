### Synchronous Value Iteration w/o and w/ Distributed Computing

Below is a review of the synchronous value iteration algorithm. The algorithm is iterative and each iteration produces a newly updated value function $V_{new}$ based on the value function from the previous iteration $V_{curr}$. This is done by applying the Bellman backup operator to $V_{curr}$ at each state. That is, 
\begin{equation}
V_{new}(s) = \max_{a\in A} R(s,a) + \beta \sum_{s'\in S} T(s,a,s') V_{curr}(s')
\end{equation}
where $\beta \in [0,1)$ is the discount factor, $R$ is the reward function, and $T$ is the transition function. 

The algorithm also maintains the greedy policy $\pi$ at each iteration, which is based on a one-step look ahead operator: 
\begin{equation}
\pi_{curr}(s) = \arg\max_{a\in A} R(s,a) + \beta \sum_{s'\in S} T(s,a,s') V_{curr}(s')
\end{equation}

After an update we define the Bellman error of that iteration as $\max_s |V_{new}(s)-V_{curr}(s)|$. In the notes, it is shown that this error allows us to bound the difference between the value function of $\pi_{curr}$ and the optimal value function $V^{*}$. Thus, a typical stopping condition for VI is to iterate until the Bellman error is below a specified threshold $\epsilon$. Putting everything together, the overall algorithm is as follows:

- Start with $V_{curr}(s) = 0$ for all $s$
- error = $\infty$
- While error > $\epsilon$ 
    - For each state $s$ 
        - $V_{new}(s) = \max_{a\in A} R(s,a) + \beta \sum_{s'\in S} T(s,a,s') V_{curr}(s')$
        - $\pi_{curr}(s) = \arg\max_{a\in A} R(s,a) + \beta \sum_{s'\in S} T(s,a,s') V_{curr}(s')$
    - error = $\max_s |V_{new}(s)-V_{curr}(s)|$   ;; could do this incrementally      
    - $V_{curr} = V_{new}$

The reason we refer to this version of VI as synchronous is because it maintains both a current and new value function, where all values of the new value function are computed based on the fixed current value function. That is, each iteration updates all states based on the value function of the previous iteration. 
