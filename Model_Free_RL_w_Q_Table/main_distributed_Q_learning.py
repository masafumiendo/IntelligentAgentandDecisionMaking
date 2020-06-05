import time
import ray

from environment import FrozenLakeEnv, generate_map, MAPS
from summarize_results import plot_result, plot_image
from Q_Learning import QLAgent
from distributed_Q_Learning import distributed_QL_agent

map_DH = (MAPS["Dangerous Hallway"], 8)
map_16 = (MAPS["16x16"], 16)

run_time = {}

def main():

    ray.shutdown()
    ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000,
             temp_dir='~/ray_tmp')

    MAP = map_DH
    map_size = MAP[1]
    simulator = FrozenLakeEnv(desc=MAP[0])

    simulator.reset()
    epsilon = 0.3  # Fix epsilon as 0.3
    learning_rate = 0.001  # Select 0.001 or 0.1
    learning_episodes = 30000  # 30000 (DH) or 100000 (16)
    test_interval = 100
    batch_size = 100
    do_test = True

    start_time = time.time()
    distributed_ql_agent = distributed_QL_agent(simulator, epsilon, learning_rate, learning_episodes, map_size,
                                                test_interval=test_interval, batch_size=batch_size,
                                                cw_num=8, ew_num=4, do_test=do_test)
    total_rewards, q_table = distributed_ql_agent.learn_and_evaluate(MAP)
    run_time['Distributed Q-learning agent'] = time.time() - start_time
    print("Learning time:\n")
    print(run_time['Distributed Q-learning agent'])
    if do_test:
        plot_result(total_rewards, test_interval, [distributed_ql_agent.agent_name])
    plot_image(q_table, MAP[0], map_size)

if __name__ == '__main__':
    main()