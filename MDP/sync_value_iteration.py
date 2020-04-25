from copy import deepcopy
import numpy as np

def sync_value_iteration(env, beta=0.999, epsilon=0.0001):
    A = env.GetActionSpace()
    S = env.GetStateSpace()

    pi = [0] * S
    v = [0] * S

    pi_new = [0] * S
    v_new = [0] * S

    error = float('inf')

    while (error > epsilon):
        error = 0
        for state in range(S):
            max_v = float('-inf')
            max_a = 0
            v_tmps = []
            for action in range(A):
                reward = env.GetReward(state, action)
                # Get successors' states with transition probabilities
                successors = env.GetSuccessors(state, action)
                expected_value = 0
                # Compute cumulative expected value
                for next_state_index in range(len(successors)):
                    expected_value += successors[next_state_index][1] * v[successors[next_state_index][0]]
                v_tmps.append(reward + beta * expected_value)
            max_v = max(v_tmps)
            max_a = v_tmps.index(max(v_tmps))
            v_new[state] = max_v
            pi_new[state] = max_a
        # Compute Bellman error
        error = max(np.abs([v_new_ - v_ for (v_new_, v_) in zip(v_new, v)]))
        v = deepcopy(v_new)
        pi = deepcopy(pi_new)
    return v, pi