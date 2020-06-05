import numpy as np
import tqdm

class agent():
    def __init__(self, simulator, epsilon, learning_rate, learning_episodes, map_size,
                 test_interval=100, action_space=4, beta=0.999, do_test=True):
        self.simulator = simulator
        self.beta = beta
        self.epsilon = epsilon
        self.test_interval = test_interval
        self.batch_num = learning_episodes // test_interval
        self.action_space = action_space
        self.state_space = map_size * map_size + 1
        self.q_table = np.zeros((self.state_space, self.action_space))
        self.learning_rate = learning_rate
        self.do_test = do_test

    def explore_or_exploit_policy(self, curr_state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            if sum(self.q_table[curr_state]) != 0:
                return self.greedy_policy(curr_state)
            else:
                return np.random.randint(self.action_space)

    def greedy_policy(self, curr_state):
        return np.argmax(self.q_table[curr_state])

    def learn_and_evaluate(self):
        total_rewards = []
        for i in tqdm.tqdm(range(self.batch_num), desc="Test Number"):
            self.learn()
            if self.do_test:
                averaged_total_reward = self.evaluate(self.greedy_policy)
                total_rewards.append(averaged_total_reward)
            else:
                pass

        return total_rewards, self.q_table

    def evaluate(self, policy_func, trials=100, max_steps=1000):

        total_reward = 0
        for _ in range(trials):
            self.simulator.reset()
            done = False
            steps = 0
            observation, reward, done, info = self.simulator.step(policy_func(0))
            total_reward += (self.beta ** steps) * reward
            steps += 1
            while not done and steps < 1000:
                observation, reward, done, info = self.simulator.step(policy_func(observation))
                total_reward += (self.beta ** steps) * reward
                steps += 1

        return total_reward / trials

    def learn(self):
        pass