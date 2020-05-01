from copy import deepcopy
import random

import numpy as np
import ray

@ray.remote
class PI_server(object):

    def __init__(self, workers_num, S, A, epsilon, beta, env):
        self.workers_num = workers_num
        self.S = S
        self.A = A
        self.epsilon = epsilon
        self.beta = beta
        self.env = env

        self.v_pi = [0] * self.S
        self.pi_current = [0] * self.S
        self.pi_new = [0] * self.S
        actions = list(range(self.A))
        for state in range(self.S):
            self.pi_current[state] = random.choice(actions)

        self.isUpdate_policy = [True] * self.S
        self.isEndEpisode = False
        self.isConvergence = False
        self.areFinished = [False] * self.workers_num

    def get_value_and_policy(self):
        return self.v_pi, self.pi_current

    def evaluate_policy(self):
        v_pi = [0] * self.S
        v_pi_new = [0] * self.S

        error = float('inf')
        while (error > self.epsilon):
            for state in range(self.S):
                reward = self.env.GetReward(state, self.pi_current[state])
                successors = self.env.GetSuccessors(state, self.pi_current[state])
                expected_value = 0
                for next_state_index in range(len(successors)):
                    expected_value += successors[next_state_index][1] * v_pi[successors[next_state_index][0]]
                v_pi_new[state] = reward + self.beta * expected_value
            error = max(np.abs([_v_pi_new - _v_pi for (_v_pi_new, _v_pi) in zip(v_pi_new, v_pi)]))
            v_pi = deepcopy(v_pi_new)

        return v_pi

    def get_value_with_stopping_condition(self, worker_index):
        isFinished = self.check_isFinished(worker_index)
        if isFinished:
            self.isEndEpisode = self.check_isEndEpisode()
            if self.isEndEpisode:
                self.update_policy()
                self.v_pi = self.evaluate_policy()
                self.isConvergence = self.check_isConvergence()
        else:
            self.areFinished[worker_index] = True

        if isFinished:
            if not self.isEndEpisode:
                return self.isEndEpisode

        return self.v_pi, self.isConvergence

    def update(self, update_indices, update_pi):
        for update_index, _update_pi in zip(update_indices, update_pi):
            self.pi_new[update_index] = _update_pi

    def update_policy(self):
        for state in range(self.S):
            if self.pi_new[state] == self.pi_current[state]:
                self.isUpdate_policy[state] = False

        self.pi_current = deepcopy(self.pi_new)

    def check_isFinished(self, worker_index):
        return self.areFinished[worker_index]

    def check_isEndEpisode(self):
        if all(self.areFinished):
            self.isEndEpisode = True
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
    S = env.GetStateSpace()
    A = env.GetActionSpace()

    isConvergence = False
    while True:
        isEndEpisode = False
        while isEndEpisode == False:
            isEndEpisode = ray.get(PI_server.get_value_with_stopping_condition.remote(worker_index))
            if not isEndEpisode == False:
                v_pi, isConvergence = isEndEpisode
                break

        if isConvergence:
            break

        update_q_pi = [0] * (end_state - start_state)
        for state in range(end_state - start_state):
            update_q_pi[state] = [0] * A
        update_pi = [0] * (end_state - start_state)
        update_indices = list(range(start_state, end_state, 1))

        for state_index, update_state in enumerate(update_indices):
            for action in range(A):
                reward = env.GetReward(update_state, action)
                successors = env.GetSuccessors(update_state, action)
                expected_value = 0
                for next_state_index in range(len(successors)):
                    expected_value += successors[next_state_index][1] * v_pi[successors[next_state_index][0]]
                update_q_pi[state_index][action] = reward + beta * expected_value

            update_pi[state_index] = update_q_pi[state_index].index(max(update_q_pi[state_index]))

        PI_server.update.remote(update_indices, update_pi)

def distributed_policy_iteration(env, beta=0.999, epsilon=0.01, workers_num=4):
    S = env.GetStateSpace()
    A = env.GetActionSpace()
    workers_list = []
    start_states = []
    end_states = []
    batch = S // workers_num
    for worker_num in range(workers_num):
        start_states.append(worker_num * batch)
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

    v_pi, pi = ray.get(_PI_server.get_value_and_policy.remote())
    return v_pi, pi