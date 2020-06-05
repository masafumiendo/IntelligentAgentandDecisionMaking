import numpy as np
import matplotlib.pyplot as plt

def plot_result(total_rewards, learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)

    plt.figure(num=1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.show()

def plot_image(q_table, MAP, map_size):
    best_value = np.max(q_table, axis=1)[:-1].reshape((map_size, map_size))
    best_policy = np.argmax(q_table, axis=1)[:-1].reshape((map_size, map_size))

    print("\n\nBest Q-value and Policy:\n")
    fig, ax = plt.subplots()
    im = ax.imshow(best_value)

    for i in range(best_value.shape[0]):
        for j in range(best_value.shape[1]):
            if MAP[i][j] in 'GH':
                arrow = MAP[i][j]
            elif best_policy[i, j] == 0:
                arrow = '<'
            elif best_policy[i, j] == 1:
                arrow = 'v'
            elif best_policy[i, j] == 2:
                arrow = '>'
            elif best_policy[i, j] == 3:
                arrow = '^'
            if MAP[i][j] in 'S':
                arrow = 'S ' + arrow
            text = ax.text(j, i, arrow,
                           ha="center", va="center", color="black")

    cbar = ax.figure.colorbar(im, ax=ax)

    fig.tight_layout()
    plt.show()