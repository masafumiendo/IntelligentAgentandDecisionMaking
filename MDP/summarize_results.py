import numpy as np
import matplotlib.pyplot as plt
import pickle
from policy_evaluation import evaluate_policy, evaluate_policy_discounted

def print_results(v, pi, map_size, env, beta, name):
    v_np, pi_np = np.array(v), np.array(pi)
    print("\nState Value:\n")
    print(np.array(v_np[:-1]).reshape((map_size, map_size)))
    print("\nPolicy:\n")
    print(np.array(pi_np[:-1]).reshape((map_size, map_size)))
    print("\nAverage reward: {}\n".format(evaluate_policy(env, pi)))
    print("Avereage discounted reward: {}\n".format(evaluate_policy_discounted(env, pi, discount_factor=beta)))
    print("State Value image view:\n")
    plt.imshow(np.array(v_np[:-1]).reshape((map_size, map_size)))
    # plt.savefig('VI_state_' + str(map_size) + 'png', bbox_inches="tight", pad_inches=0.05)
    plt.show()

    # pickle.dump(v, open(name + "_" + str(map_size) + "_v.pkl", "wb"))
    # pickle.dump(pi, open(name + "_" + str(map_size) + "_pi.pkl", "wb"))