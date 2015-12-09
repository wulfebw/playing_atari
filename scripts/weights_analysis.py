import numpy as np
import os

def moving_average(array, over):
    averages = []
    for x in xrange(len(array)):
        base_index = x - over
        if base_index < 0:
            base_index = 0
        vals = array[base_index:x]
        averages.append(np.mean(vals))
    return averages

def plotting_scratch_work():
    plt.plot(np.arange(len(lin_ars)), lin_ars, color='b', label='Q-learning linear')
    plt.plot(np.arange(len(rand_ars)), rand_ars, color='g', label='Random Baseline')
    plt.plot(np.arange(len(nnet_ars)), nnet_ars, color='r', label='Q-learning neural network')
    plt.xlabel('Episodes')
    plt.ylabel('Adjusted Score')
    plt.legend(loc='upper left')
    plt.show()