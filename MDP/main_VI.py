import time
from environment import FrozenLakeEnv, generate_map, MAPS
from summarize_results import print_results
from sync_value_iteration import sync_value_iteration
from sync_distributed_value_iteration import distribured_value_iteraion

import numpy as np
import ray

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
    v, pi = sync_value_iteration(env, beta=beta)
    v_np, pi_np = np.array(v), np.array(pi)
    end_time = time.time()
    run_time['Sync Value Iteration'] = end_time - start_time
    print("time:", run_time['Sync Value Iteration'])

    print_results(v, pi, map_size, env, beta, 'sync_vi_gs')

    ray.shutdown()
    ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

    beta = 0.999
    env = FrozenLakeEnv(desc=MAP[0], is_slippery=True)
    print("Game Map:")
    env.render()

    start_time = time.time()
    v, pi = distribured_value_iteraion(env, beta=beta, workers_num=4)
    v_np, pi_np = np.array(v), np.array(pi)
    end_time = time.time()
    run_time['Sync distributed VI'] = end_time - start_time
    print("time:", run_time['Sync distributed VI'])
    print_results(v, pi, map_size, env, beta, 'dist_vi')

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