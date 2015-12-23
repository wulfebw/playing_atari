"""
The SARSALambdaAgent class wraps a deep SARSA network
"""

import os
import cPickle
import time
import logging

import numpy as np

import ale_data_set

import sys
sys.setrecursionlimit(10000)

class SARSALambdaAgent(object):

    def __init__(self, sarsa_network, epsilon_start, epsilon_min,
                 epsilon_decay, exp_pref, rng):

        self.network = sarsa_network
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.exp_pref = exp_pref
        self.rng = rng
        self.phi_length = self.network.num_frames
        self.image_height = self.network.input_height
        self.image_width = self.network.input_width
        self.data_set = ale_data_set.DataSet(self.image_height, self.image_width, self.phi_length, rng)

        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_dir = self.exp_pref + time_str + \
                       "{}".format(self.network.lr).replace(".", "p") + "_" \
                       + "{}".format(self.network.discount).replace(".", "p")

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self.num_actions = self.network.num_actions

        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self._open_results_file()
        self._open_learning_file()

        self.episode_counter = 0

        self.last_img = None
        self.last_action = None

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """
        self.step_counter = 0
        self.episode_reward = 0
        self.loss_averages = []
        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)
        self.last_action = return_action
        self.last_img = observation
        return return_action

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """
        self.step_counter += 1
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)

        reward = np.clip(reward, -1, 1)
        self.data_set.add_sample(self.last_img, self.last_action, reward)

        action = self._choose_action(elf.epsilon, observation, reward)
        
        loss = self._do_training()
        self.loss_averages.append(loss)
        self.last_action = action
        self.last_img = observation
        return action

    def _choose_action(self, epsilon, cur_img, reward):
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            action = self.network.choose_action(phi, epsilon)
        else:
            action = self.rng.randint(0, self.num_actions)
        return action

    def _do_training(self):
        state, action, reward, next_state, next_action = self.data_set.get_training_tuple()
        return self.network.train(state, action, reward, next_state, next_action)

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """
        self.episode_reward += reward
        self.step_counter += 1
        total_time = time.time() - self.start_time

        self.data_set.add_sample(self.last_img, self.last_action, np.clip(reward, -1, 1), True)

        logging.info("steps/second: {:.2f}".format(self.step_counter/total_time))

        self._update_learning_file()
        logging.info("average loss: {:.4f}".format(np.mean(self.loss_averages)))

    def finish_epoch(self, epoch):
        net_file = open(self.exp_dir + '/network_file_' + str(epoch) + '.pkl', 'w')
        cPickle.dump(self.network, net_file, -1)
        net_file.close()

    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes):
        out = "{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                            self.total_reward / float(num_episodes))
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        out = "{},{}\n".format(np.mean(self.loss_averages), self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

if __name__ == "__main__":
    pass