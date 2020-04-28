from copy import deepcopy
import random

import numpy as np
import ray

@ray.remote
class PI_server(object):

    def __init__(self, workers_num, S, A, epsilon, beta, env):
        self.S = S
        self.A = A
        self.v_current = [0] * self.S
        self.pi_new = [0] * self.S
        self.isUpdate_policy = [True] * self.S
        self.workers_num = workers_num
        self.isEndEpisode = False
        self.isConvergence = False
        self.areFinished = [False] * workers_num
        self.epsilon = epsilon
        self.beta = beta
        self.env = env

        self.pi = [0] * self.S
        self.actions = list(range(self.A))
        for state_index in self.pi:
            self.pi[state_index] = random.choice(self.actions)

        self.q_pi = [0] * self.S
        for state in range(self.S):
            self.q_pi[state] = [0] * self.A

    def initialize_q_pi(self):
        self.q_pi = [0] * self.S
        for state in range(self.S):
            self.q_pi[state] = [0] * self.A

    def get_policy(self):
        return self.pi

    def get_value_and_policy(self):
        return self.evaluate_policy(), self.pi

    def get_value_with_stopping_condition(self, worker_index):
        isFinished = self.check_isFinished(worker_index)
        if isFinished:
            self.isEndEpisode = self.check_isEndEpisode()
            if self.isEndEpisode:
                self.check_policy_and_update()
                self.isConvergence = self.check_isConvergence()
        else:
            self.areFinished[worker_index] = True

        if isFinished:
            if not self.isEndEpisode:
                return self.isEndEpisode

        return self.evaluate_policy(), self.isConvergence

    def evaluate_policy(self):
        v_pi = [0] * self.S
        v_pi_new = [0] * self.S

        pi = self.get_policy()

        error = float('inf')

        while (error > self.epsilon):
            error = 0
            for state in range(self.S):
                reward = self.env.GetReward(state, pi[state])
                # Get successors' states with transition probabilities
                successors = self.env.GetSuccessors(state, pi[state])
                expected_value = 0
                # Compute cumulative expected value
                for next_state_index in range(len(successors)):
                    expected_value += successors[next_state_index][1] * v_pi[successors[next_state_index][0]]
                v_pi_new[state] = reward + self.beta * expected_value
            # Compute Bellman error
            error = max(np.abs([_v_pi_new - _v_pi for (_v_pi_new, _v_pi) in zip(v_pi_new, v_pi)]))
            v_pi = deepcopy(v_pi_new)
        return v_pi

    def update(self, update_indices, update_q_pis):
        for update_index, update_q_pi in zip(update_indices, update_q_pis):
            self.q_pi[update_index] = update_q_pi

    def check_policy_and_update(self):
        for state in range(self.S):
            self.pi_new[state] = self.q_pi[state].index(max(self.q_pi[state]))
            if self.pi_new[state] == self.pi[state]:
                self.isUpdate_policy[state] = False

        self.pi = deepcopy(self.pi_new)

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
        if not any(self.isUpdate_policy):
            isConvergence = True
        else:
            isConvergence = False
        return isConvergence

@ray.remote
def PI_worker(worker_index, PI_server, data, start_state, end_state):
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

        update_q_pis = [0] * (end_state - start_state)
        for state in range(end_state - start_state):
            update_q_pis[state] = [0] * A
        update_indices = list(range(start_state, end_state, 1))

        for state_index, update_state in enumerate(update_indices):
            for action in range(A):
                reward = env.GetReward(update_state, action)
                successors = env.GetSuccessors(update_state, action)
                expected_value = 0
                for next_state_index in range(len(successors)):
                    expected_value += successors[next_state_index][1] * V[successors[next_state_index][0]]
                update_q_pis[state_index][action] = reward + beta * expected_value

        PI_server.update.remote(update_indices, update_q_pis)

def distributed_policy_iteration(env, beta=0.999, epsilon=0.01, workers_num=4):
    S = env.GetStateSpace()
    A = env.GetActionSpace()
    workers_list = []
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

    _PI_server = PI_server.remote(workers_num, S, A, epsilon, beta, env)
    data_id = ray.put((env, workers_num, beta, epsilon))

    w_ids = []
    for worker_index in range(workers_num):
        w_id = PI_worker.remote(worker_index, _PI_server, data_id, start_states[worker_index], end_states[worker_index])
        w_ids.append(w_id)
    ray.wait(w_ids, num_returns=workers_num, timeout=None)

    v, pi = ray.get(_PI_server.get_value_and_policy.remote())
    return v, pi