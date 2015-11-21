"""
:description: multilayer perceptron and hidden layer classes
"""

import numpy as np
import theano
import theano.tensor as T

class MLP(object):

	def __init__(self, layers):
		self.layers = layers
		self.cost = cost

	def fprop(self, input):
		state_below = input
		outputs = []
		for layer in self.layers:
			state_below = layer.fprop(state_below)
			outputs.append(state_below)
		return outputs[-1]

	def get_cost_updates(self, prediction, target, learning_rate=0.01):
		cost = T.mean(T.sqr(targets - prediction))
		params = self.get_params()
		gparams = T.grad(cost, params)
		updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]
		return (cost, updates)

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.params
		return params

class Hiddenlayer(object):

	def __init__(self, n_vis, n_hid, layer_name, rng=None, param_init_range=0.02):
		if rng is None:
			rng = np.random.RandomState()
		self.rng = rng
		self.n_vis = n_vis
		self.n_hid = n_hid
		self.layer_name = layer_name
		self.param_init_range = param_init_range
		self.activation = T.sigmoid

		# input-to-hidden (rows, cols) = (n_visible, n_hidden)
		init_W = rng.uniform(-self.param_init_range, self.param_init_range, (self.n_vis, self.n_hid))
		self.W = theano.shared(value=init_W, name=self.layer_name + '_W', borrow=True)
		self.b = theano.shared(value=np.zeros(self.n_hid), name=self.layer_name + '_b', borrow=True)

		self.params = [self.W, self.b]

	def fprop(self, state_below):
		return self.activation(T.dot(state_below, self.W) + self.b)
		