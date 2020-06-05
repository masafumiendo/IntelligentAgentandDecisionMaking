import numpy as np

import ray

from TD_agent import agent
from environment import FrozenLakeEnv, generate_map, MAPS

@ray.remote
class QLAgent_server(agent):
    def __init__(self, simulator, epsilon, learning_rate, learning_episodes, map_size,
                 test_interval=100, batch_size=100, action_space=4, beta=0.999, do_test=True):
        super().__init__(simulator, epsilon, learning_rate, learning_episodes, map_size,
                         test_interval=test_interval, action_space=action_space, beta=beta, do_test=do_test)
        self.collector_done = False
        self.evaluator_done = False
        self.learning_episodes = learning_episodes
        self.episode = 0
        self.reuslts = []
        self.batch_size = batch_size
        self.privous_q_tables = []
        self.results = [0] * (self.batch_num + 1)
        self.reuslt_count = 0

    def learn(self, experiences):

        if not self.collector_done:
            for experience in experiences:
                observation, action, reward, observation_nxt = experience  # Retrieve list of (s, a, r, s')
                estimate = self.q_table[observation][action]
                gain = reward + self.beta * max(self.q_table[observation_nxt])  # Off-policy
                self.q_table[observation][action] += self.learning_rate * (gain - estimate)
            self.episode += self.batch_size

        if self.do_test:
            if self.episode // self.test_interval + 1 > len(self.privous_q_tables):
                self.privous_q_tables.append(self.q_table)
        return self.get_q_table()

    def get_q_table(self):
        if self.episode >= self.learning_episodes:
            self.collector_done = True

        return self.q_table, self.collector_done

    # evaluator
    def add_result(self, result, num):
        self.results[num] = result

    def get_reuslts(self):
        return self.results, self.q_table

    def ask_evaluation(self):
        if len(self.privous_q_tables) > self.reuslt_count:
            num = self.reuslt_count
            evluation_q_table = self.privous_q_tables[num]
            self.reuslt_count += 1
            return evluation_q_table, False, num
        else:
            if self.episode >= self.learning_episodes:
                self.evaluator_done = True
            return [], self.evaluator_done, None


@ray.remote
def collecting_worker(server, simulator, epsilon, action_space=4, batch_size=100):
    def greedy_policy(q_table, curr_state):
        return np.argmax(q_table[curr_state])

    def explore_or_exploit_policy(q_table, curr_state):
        if np.random.random() < epsilon:
            return np.random.randint(action_space)
        else:
            if sum(q_table[curr_state]) != 0:
                return greedy_policy(q_table, curr_state)
            else:
                return np.random.randint(action_space)

    q_table, learn_done = ray.get(server.get_q_table.remote())
    while True:
        if learn_done:
            break
        else:
            experiences = []
            for _ in range(batch_size):
                observation = simulator.reset()
                done = False
                while not done:
                    action = explore_or_exploit_policy(q_table, observation)
                    observation_nxt, reward, done, info = simulator.step(action)
                    experiences.append([observation, action, reward, observation_nxt])  # list of (s, a, r, s')
                    observation = observation_nxt

            q_table, learn_done = ray.get(server.learn.remote(experiences))  # Send experiences to the server


@ray.remote
def evaluation_worker(server, simulator, trials=100, action_space=4, beta=0.999):
    def greedy_policy(q_table, curr_state):
        return np.argmax(q_table[curr_state])

    while True:
        q_table, done, num = ray.get(server.ask_evaluation.remote())  # Take Q-table from the server
        if done:
            break
        if len(q_table) == 0:
            continue
        total_reward = 0
        for _ in range(trials):
            simulator.reset()
            done = False
            steps = 0
            observation, reward, done, info = simulator.step(greedy_policy(q_table, 0))
            total_reward += (beta ** steps) * reward
            steps += 1
            while not done:
                observation, reward, done, info = simulator.step(greedy_policy(q_table, observation))
                total_reward += (beta ** steps) * reward
                steps += 1
        server.add_result.remote(total_reward / trials, num)


class distributed_QL_agent():
    def __init__(self, simulator, epsilon, learning_rate, learning_episodes, map_size,
                 cw_num=4, ew_num=4, test_interval=100, batch_size=100,
                 action_space=4, beta=0.999, do_test=True):

        self.server = QLAgent_server.remote(simulator, epsilon, learning_rate, learning_episodes, map_size,
                                            test_interval=test_interval, batch_size=batch_size, do_test=do_test)
        self.workers_id = []
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.cw_num = cw_num  # Number of collectors
        self.ew_num = ew_num  # Number of evaluators
        self.agent_name = "Distributed Q-learning"
        self.do_test = do_test

    def learn_and_evaluate(self, MAP):
        workers_id = []

        for _ in range(self.cw_num):
            simulator = FrozenLakeEnv(desc=MAP[0])
            collecting_worker.remote(self.server, simulator, self.epsilon)
        for _ in range(self.ew_num):
            worker_id = evaluation_worker.remote(self.server, simulator)
            workers_id.append(worker_id)

        ray.wait(workers_id, len(workers_id))
        return ray.get(self.server.get_reuslts.remote())