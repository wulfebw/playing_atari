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


qlin = moving_average(np.load(open('latest_first_run_linear.npz'))['rewards'], 100)[:2200]
sar0 = moving_average(np.load(open('sarsa0rewards.npz'))['rewards'], 100)[:2200]
sar50 = moving_average(np.load(open('sarsa50rewards.npz'))['rewards'], 100)[:2200]
sar98 = moving_average(np.load(open('sarsa98rewards.npz'))['rewards'], 100)[:2200]
rand = np.load(open('random_latest.npz'))['rewards']
nrand = []
for x in xrange(2200):
    nrand.append(random.choice(rand))

nrand = moving_average(nrand, 100)
rand = nrand


plt.plot(qlin, color='b', label='replay memory q-learning')
plt.plot(sar0, color='c', label='sarsa lambda = 0.0')
plt.plot(sar50, color='r', label='sarsa lambda = 0.50')
plt.plot(sar98, color='g', label='sarsa lambda = 0.98')
plt.plot(rand, color='k', label='random')
plt.xlabel('Episodes')
plt.ylabel('Adjusted Score')
plt.legend(loc='upper left')


def plotting_scratch_work():
    plt.plot(np.arange(len(lin_ars)), lin_ars, color='b', label='Q-learning linear')
    plt.plot(np.arange(len(rand_ars)), rand_ars, color='g', label='Random Baseline')
    plt.plot(np.arange(len(nnet_ars)), nnet_ars, color='r', label='Q-learning neural network')
    plt.xlabel('Episodes')
    plt.ylabel('Adjusted Score')
    plt.legend(loc='upper left')
    plt.show()