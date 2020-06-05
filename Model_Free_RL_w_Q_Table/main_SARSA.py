import time
from environment import FrozenLakeEnv, generate_map, MAPS
from summarize_results import plot_result, plot_image
from SARSA import SARSAagent

map_DH = (MAPS["Dangerous Hallway"], 8)
map_16 = (MAPS["16x16"], 16)

run_time = {}

def main():

    MAP = map_DH
    map_size = MAP[1]
    simulator = FrozenLakeEnv(desc=MAP[0])

    epsilon = 0.3  # Select 0.3 or 0.05
    learning_rate = 0.001  # Select 0.001 or 0.1
    learning_episodes = 30000  # 30000 (DH) or 100000 (16)
    test_interval = 100
    do_test = True

    sarsa_agent = SARSAagent(simulator, epsilon, learning_rate, learning_episodes, map_size,
                       test_interval=test_interval, do_test=do_test)
    start_time = time.time()
    total_rewards, q_table = sarsa_agent.learn_and_evaluate()
    run_time['SARSA agent'] = time.time() - start_time
    print("Learning time:\n")
    print(run_time['SARSA agent'])
    if do_test:
        plot_result(total_rewards, sarsa_agent.test_interval, [sarsa_agent.agent_name])
    plot_image(q_table, MAP[0], map_size)

if __name__ == '__main__':
    main()