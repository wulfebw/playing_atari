
import numpy as np
import theano

floatX = theano.config.floatX

class DataSet(object):

    def __init__(self, height, width, phi_length, rng):
        self.height = height
        self.width = width
        self.phi_length = phi_length
        self.max_steps = phi_length * 2
        self.rng = rng

        self.imgs = np.zeros((self.max_steps, height, width), dtype='uint8')
        self.actions = np.zeros(self.max_steps, dtype='int32')
        self.rewards = np.zeros(self.max_steps, dtype=floatX)
       
    def add_sample(self, img, action, reward):
        self.imgs[self.top] = img
        self.actions[self.top] = action
        self.rewards[self.top] = reward

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps 
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def phi(self, img):
        indexes = np.arange(self.top - self.phi_length + 1, self.top)
        phi = np.empty((self.phi_length, self.height, self.width), dtype=floatX)
        phi[0:self.phi_length - 1] = self.imgs.take(indexes, axis=0, mode='wrap')
        phi[-1] = img
        return phi

    def get_training_tuple(self):
        next_indexes = np.arange(self.top - self.phi_length, self.top)
        next_state = self.imgs.take(next_indexes, axis=0, mode='wrap')
        next_action = self.actions[next_indexes[-1]]

        cur_indexes = next_indexes - 1
        state = self.imgs.take(cur_indexes, axis=0, mode='wrap')
        action = self.actions[cur_indexes[-1]]
       
        reward = self.rewards[cur_indexes[-1]]

        return state, action, reward, next_state, next_action