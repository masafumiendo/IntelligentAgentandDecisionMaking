import time
from environment import FrozenLakeEnv, generate_map, MAPS
from summarize_results import print_results
from sync_policy_iteration import sync_policy_iteration

import numpy as np

map_8 = (MAPS["8x8"], 8)
map_16 = (MAPS["16x16"], 16)
map_32 = (MAPS["32x32"], 32)
map_50 = (generate_map((50,50)), 50)
map_110 = (generate_map((110,110)), 110)

run_time = {}

def main():

    MAP = map_8
    map_size = MAP[1]

    beta = 0.999
    env = FrozenLakeEnv(desc=MAP[0], is_slippery=True)
    print("Game Map:")
    env.render()

    start_time = time.time()
    v, pi = sync_policy_iteration(env, beta=beta)
    v_np, pi_np = np.array(v), np.array(pi)
    end_time = time.time()
    run_time['Sync Policy Iteration'] = end_time - start_time
    print("time:", run_time['Sync Policy Iteration'])

    print_results(v, pi, map_size, env, beta, 'sync_pi_gs')

    from copy import deepcopy
    temp_dict = deepcopy(run_time)
    print("All:")
    for _ in range(len(temp_dict)):
        min_v = float('inf')
        for k, v in temp_dict.items():
            if v is None:
                continue
            if v < min_v:
                min_v = v
                name = k
        temp_dict[name] = float('inf')
        print(name + ": " + str(min_v))
        print()

if __name__ == '__main__':
    main()