from TD_agent import agent

class SARSAagent(agent):
    def __init__(self, simulator, epsilon, learning_rate, learning_episodes, map_size,
                 test_interval=100, action_space=4, beta=0.999, do_test=True):
        super().__init__(simulator, epsilon, learning_rate, learning_episodes, map_size,
                         test_interval=test_interval, action_space=action_space, beta=beta, do_test=do_test)
        self.agent_name = "SARSA Agent"

    def learn(self):
        for _ in range(self.test_interval):

            observation = self.simulator.reset()
            done = False
            while not done:
                action = self.explore_or_exploit_policy(observation)
                observation_nxt, reward, done, info = self.simulator.step(action)
                action_nxt = self.explore_or_exploit_policy(observation_nxt)
                estimate = self.q_table[observation][action]
                gain = reward + self.beta * self.q_table[observation_nxt][action_nxt]  # On-policy
                self.q_table[observation][action] += self.learning_rate * (gain - estimate)
                observation = observation_nxt
