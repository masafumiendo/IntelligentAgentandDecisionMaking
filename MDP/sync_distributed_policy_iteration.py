from copy import deepcopy
import itertools
import random

import numpy as np
import ray

@ray.remote
class PI_server(object):

    def __init__(self, workers_num, start_states, end_states, epsilon):
        self.v_current_per_worker = []
        self.pi_per_worker = []
        self.v_new_per_worker = []
        for start_state, end_state in zip(start_states, end_states):
            self.v_current_per_worker.append([0] * (end_state - start_state))
            self.pi_per_worker.append([0] * (end_state - start_state))
            self.v_new_per_worker.append([0] * (end_state - start_state))
        self.pi_new_per_worker = deepcopy(self.pi_per_worker)
        self.workers_num = workers_num
        self.isEndEpisode = False
        self.isConvergence = False
        self.areFinished = [False] * workers_num
        self.epsilon = epsilon

    def get_value(self):
        return list(itertools.chain.from_iterable(self.v_current_per_worker))

    def get_value_and_policy(self):
        v_current = list(itertools.chain.from_iterable(self.v_current_per_worker))
        pi = list(itertools.chain.from_iterable(self.pi_per_worker))
        return v_current, pi

    def get_value_with_stopping_condition(self, worker_index):
        isFinished = self.check_isFinished(worker_index)
        if isFinished:
            self.isEndEpisode = self.check_isEndEpisode()
            if self.isEndEpisode:
                self.get_error_and_update(self.workers_num)
                self.isConvergence = self.check_isConvergence()
        else:
            self.areFinished[worker_index] = True

        if isFinished:
            if not self.isEndEpisode:
                return self.isEndEpisode

        return self.get_value(), self.isConvergence

    def update(self, worker_index, update_vs):
        self.v_new_per_worker[worker_index] = update_vs

    def get_error_and_update(self, workers_num):
        self.max_error = 0
        for worker_index in range(workers_num):
            errors = (np.array(self.v_new_per_worker[worker_index]) -
                      np.array(self.v_current_per_worker[worker_index])).tolist()
            error = max(np.abs(errors))
            if error > self.max_error:
                self.max_error = error

        self.v_current_per_worker = deepcopy(self.v_new_per_worker)

    def check_isFinished(self, worker_index):
        return self.areFinished[worker_index]

    def check_isEndEpisode(self):
        if all(self.areFinished):
            self.isEndEpisode = True
            # Reset
            self.areFinished = [False] * self.workers_num
        else:
            self.isEndEpisode = False
        return self.isEndEpisode

    def check_isConvergence(self):
        if self.max_error < self.epsilon:
            isConvergence = True
        else:
            isConvergence = False
        return isConvergence

@ray.remote
def PI_worker(worker_index, PI_server, data, start_state, end_state, pi):
    env, workers_num, beta, epsilon = data
    A = env.GetActionSpace()
    S = env.GetStateSpace()

    isConvergence = False
    while True:
        isEndEpisode = False
        while isEndEpisode == False:
            isEndEpisode = ray.get(PI_server.get_value_with_stopping_condition.remote(worker_index))
            if not isEndEpisode == False:
                V, isConvergence = isEndEpisode
                break

        if isConvergence:
            break

        update_vs = [0] * (end_state - start_state)
        update_indices = list(range(start_state, end_state, 1))

        for state_index, update_state in enumerate(update_indices):
            v_tmps = []

            reward = env.GetReward(update_state, pi[update_state])
            # Get successors' states with transition probabilities
            successors = env.GetSuccessors(update_state, pi[update_state])
            expected_value = 0
            # Compute cumulative expected value
            for next_state_index in range(len(successors)):
                expected_value += successors[next_state_index][1] * V[successors[next_state_index][0]]
            update_vs[state_index] = reward + beta * expected_value

        PI_server.update.remote(worker_index, update_vs)

def initialize_policy(S, A):
    actions = list(range(A))
    pi = [0] * S
    for state_index in pi:
        pi[state_index] = random.choice(actions)
    return pi

def initialize_q_pi(A, S):
    q_pi = [0] * S
    for state in range(S):
        q_pi[state] = [0] * A
    return q_pi

def policy_evaluation(env, pi, beta=0.999, epsilon=0.01, workers_num=4):
    S = env.GetStateSpace()
    start_states = []
    end_states = []
    batch = S // workers_num
    for worker_num in range(workers_num):
        start_states.append(worker_num * batch)
        # If statement to deal with residual states
        if worker_num == workers_num - 1:
            end_states.append(S)
        else:
            end_states.append((worker_num + 1) * batch)

    _PI_server = PI_server.remote(workers_num, start_states, end_states, epsilon)
    data_id = ray.put((env, workers_num, beta, epsilon))

    w_ids = []
    for worker_index in range(workers_num):
        w_id = PI_worker.remote(worker_index, _PI_server, data_id, start_states[worker_index],
                                end_states[worker_index], pi)
        w_ids.append(w_id)
    ray.wait(w_ids, num_returns=workers_num, timeout=None)

    v, pi = ray.get(_PI_server.get_value_and_policy.remote())
    return v

def distributed_policy_iteration(env, beta=0.999, epsilon=0.01, workers_num=4):
    A = env.GetActionSpace()
    S = env.GetStateSpace()

    pi = initialize_policy(S, A)
    isUpdate_policy = [True] * S

    pi_new = deepcopy(pi)

    while True:
        q_pi = initialize_q_pi(A, S)
        v = policy_evaluation(env, pi)
        for state in range(S):
            for action in range(A):
                reward = env.GetReward(state, action)
                successors = env.GetSuccessors(state, action)
                expected_value = 0
                for next_state_index in range(len(successors)):
                    expected_value += successors[next_state_index][1] * v[successors[next_state_index][0]]
                q_pi[state][action] = reward + beta * expected_value
            pi_new[state] = q_pi[state].index(max(q_pi[state]))

            if pi_new[state] == pi[state]:
                isUpdate_policy[state] = False
        pi = deepcopy(pi_new)
        # If there are no improvements for policy, the iteration process will be finished
        if not any(isUpdate_policy):
            break

    return v, pi