from copy import deepcopy
import itertools

import numpy as np
import ray

@ray.remote
class VI_server(object):

    def __init__(self, workers_num, S, epsilon):
        self.v_current = [0] * S
        self.pi = [0] * S
        self.v_new = [0] * S
        self.workers_num = workers_num
        self.isEndEpisode = False
        self.isConvergence = False
        self.areFinished = [False] * workers_num
        self.epsilon = epsilon

    def get_value(self):
        return self.v_current

    def get_value_and_policy(self):
        return self.v_current, self.pi

    def get_value_with_stopping_condition(self, worker_index):
        isFinished = self.check_isFinished(worker_index)
        if isFinished:
            self.isEndEpisode = self.check_isEndEpisode()
            if self.isEndEpisode:
                self.get_error_and_update()
                self.isConvergence = self.check_isConvergence()
        else:
            self.areFinished[worker_index] = True

        if isFinished:
            if not self.isEndEpisode:
                return self.isEndEpisode

        return self.get_value(), self.isConvergence

    def update(self, update_indices, update_vs, update_pis):
        for update_index, update_v, update_pi in zip(update_indices, update_vs, update_pis):
            self.v_new[update_index] = update_v
            self.pi[update_index] = update_pi

    def get_error_and_update(self):
        self.max_error = 0
        errors = np.array(self.v_new) - np.array(self.v_current).tolist()
        error = max(np.abs(errors))
        if error > self.max_error:
            self.max_error = error

        self.v_current = deepcopy(self.v_new)

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
def VI_worker(worker_index, VI_server, data, start_state, end_state):
    env, workers_num, beta, epsilon = data
    A = env.GetActionSpace()
    S = env.GetStateSpace()

    isConvergence = False
    while True:
        isEndEpisode = False
        while isEndEpisode == False:
            isEndEpisode = ray.get(VI_server.get_value_with_stopping_condition.remote(worker_index))
            if not isEndEpisode == False:
                V, isConvergence = isEndEpisode
                break

        if isConvergence:
            break

        update_pis = [0] * (end_state - start_state)
        update_vs = [0] * (end_state - start_state)
        update_indices = list(range(start_state, end_state, 1))

        for state_index, update_state in enumerate(update_indices):
            v_tmps = []
            for action in range(A):
                reward = env.GetReward(update_state, action)
                successors = env.GetSuccessors(update_state, action)
                expected_value = 0
                for next_state_index in range(len(successors)):
                    expected_value += successors[next_state_index][1] * V[successors[next_state_index][0]]
                v_tmps.append(reward + beta * expected_value)
            update_vs[state_index] = max(v_tmps)
            update_pis[state_index] = v_tmps.index(max(v_tmps))

        VI_server.update.remote(update_indices, update_vs, update_pis)

def distribured_value_iteraion(env, beta=0.999, epsilon=0.01, workers_num=4):
    S = env.GetStateSpace()
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

    _VI_server = VI_server.remote(workers_num, S, epsilon)
    data_id = ray.put((env, workers_num, beta, epsilon))

    w_ids = []
    for worker_index in range(workers_num):
        w_id = VI_worker.remote(worker_index, _VI_server, data_id, start_states[worker_index], end_states[worker_index])
        w_ids.append(w_id)
    ray.wait(w_ids, num_returns=workers_num, timeout=None)

    v, pi = ray.get(_VI_server.get_value_and_policy.remote())
    return v, pi