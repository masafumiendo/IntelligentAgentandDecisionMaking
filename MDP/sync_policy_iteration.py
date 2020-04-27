from copy import deepcopy
import numpy as np
import random

def initialize_policy(S):
    actions = [0, 1, 2, 3] # hard-coded
    pi = [0] * S
    for state_index in pi:
        pi[state_index] = random.choice(actions)
    return pi

def initialize_q_pi(A, S):
    q_pi = [0] * S
    for state in range(S):
        q_pi[state] = [0] * A
    return q_pi

def policy_evaluation(env, pi, beta=0.999, epsilon=0.0001):
    S = env.GetStateSpace()

    v = [0] * S
    v_new = [0] * S

    error = float('inf')

    while (error > epsilon):
        error = 0
        for state in range(S):
            reward = env.GetReward(state, pi[state])
            # Get successors' states with transition probabilities
            successors = env.GetSuccessors(state, pi[state])
            expected_value = 0
            # Compute cumulative expected value
            for next_state_index in range(len(successors)):
                expected_value += successors[next_state_index][1] * v[successors[next_state_index][0]]
            v_new[state] = reward + beta * expected_value
        # Compute Bellman error
        error = max(np.abs([v_new_ - v_ for (v_new_, v_) in zip(v_new, v)]))
        v = deepcopy(v_new)
    return v

def sync_policy_iteration(env, beta=0.999, epsilon=0.0001):
    A = env.GetActionSpace()
    S = env.GetStateSpace()

    pi = initialize_policy(S)
    isUpdate_policy = [True] * S

    pi_new = deepcopy(pi)

    while True:
        q_pi = initialize_q_pi(A, S)
        v_pi = policy_evaluation(env, pi)
        for state in range(S):
            for action in range(A):
                reward = env.GetReward(state, action)
                successors = env.GetSuccessors(state, action)
                expected_value = 0
                for next_state_index in range(len(successors)):
                    expected_value += successors[next_state_index][1] * v_pi[successors[next_state_index][0]]
                q_pi[state][action] = reward + beta * expected_value
            pi_new[state] = q_pi[state].index(max(q_pi[state]))

            if pi_new[state] == pi[state]:
                isUpdate_policy[state] = False
        pi = deepcopy(pi_new)
        # If there are no improvements for policy, the iteration process will be finished
        if not any(isUpdate_policy):
            break
    return v_pi, pi