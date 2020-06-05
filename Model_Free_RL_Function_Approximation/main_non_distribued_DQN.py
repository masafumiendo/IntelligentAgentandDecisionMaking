import gym

from non_distributed_DQN_agent import DQN_agent
from plot_result import plot_result

ENV_NAME = 'CartPole-v0'

hyperparams_CartPole = {
    'epsilon_decay_steps' : 100000,
    'final_epsilon' : 0.1,
    'batch_size' : 32,
    'update_steps' : 10,
    'memory_size' : 2000,
    'beta' : 0.99,
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True
}

def main():

    env_CartPole = gym.make(ENV_NAME)

    training_episodes, test_interval = 10000, 50
    agent = DQN_agent(env_CartPole, hyperparams_CartPole)
    result = agent.learn_and_evaluate(training_episodes, test_interval)
    plot_result(result, test_interval, ["batch_update with target_model"])

if __name__ == '__main__':
    main()