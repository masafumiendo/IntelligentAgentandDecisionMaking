import time
import ray

from custom_cartpole import CartPoleEnv
from distributed_DQN_agent import distributed_DQN_agent
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

    ray.shutdown()
    ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

    env = CartPoleEnv()
    env.reset()

    cw_num = 8
    ew_num = 8
    training_episodes, test_interval, trials = 10000, 50, 30
    agent = distributed_DQN_agent(env, hyperparams_CartPole, cw_num, ew_num, training_episodes, test_interval, trials)
    start_time = time.time()
    result = agent.learn_and_evaluate()
    run_time = {}
    run_time['distributed DQN agent'] = time.time() - start_time
    print("running time: ", run_time['distributed DQN agent'])

    plot_result(result, test_interval, ["batch_update with target_model"])

if __name__ == '__main__':
    main()