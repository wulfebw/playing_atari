"""
:description: multilayer perceptron and hidden layer classes
"""

import numpy as np
import theano
import theano.tensor as T

class MLP(object):
	"""
	:description: the MLP class acts as a wrapper around the layers of the network.
	"""
	def __init__(self, layers, discount, learning_rate):
		"""
		:type layers: list of objects
		:param layers: list of the layers (in order) of this mlp
		"""
		self.layers = layers
		self.discount = discount
		self.learning_rate = learning_rate

	def fprop(self, features):
		"""
		:description: forward propagates the input through all layers of the network

		:type input: theano.tensor.col
		:param input: the original input features to the network as a 1 dimensional col vector

		:type rval: theano.tensor.col
		:param rval: the output value of the entire network (i.e., the Q values for each action). 
		"""
		outputs = []
		for layer in self.layers:
			features = layer.fprop(features)
			outputs.append(features)
		return outputs[-1]

	def get_loss_and_updates(self, features, action, reward, next_features):
		q_values = self.fprop(features)
		next_q_values = self.fprop(next_features)
		target = reward + self.discount * T.max(next_q_values)
		loss = .5 * T.sqr(target - q_values[action])

		params = self.get_params()
		gparams = T.grad(loss, params)
		updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(params, gparams)]
		return (loss, updates)

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.params
		return params

class HiddenLayer(object):

	def __init__(self, n_vis, n_hid, layer_name, rng=None, param_init_range=0.02):
		if rng is None:
			rng = np.random.RandomState()
		self.rng = rng
		self.n_vis = n_vis
		self.n_hid = n_hid
		self.layer_name = layer_name
		self.param_init_range = param_init_range
		self.activation = theano.tensor.nnet.sigmoid

		# input-to-hidden (rows, cols) = (n_visible, n_hidden)
		init_W = rng.uniform(-self.param_init_range, self.param_init_range, (self.n_vis, self.n_hid))
		self.W = theano.shared(value=init_W, name=self.layer_name + '_W', borrow=True)
		self.b = theano.shared(value=np.zeros(self.n_hid), name=self.layer_name + '_b', borrow=True)

		self.params = [self.W, self.b]

	def fprop(self, state_below):
		return self.activation(T.dot(state_below, self.W) + self.b)

class OutputLayer(object):

	def __init__(self, layer_name, alpha=0.5):
		self.layer_name = layer_name
		self.params = []
		self.alpha = alpha

	def fprop(self, state_below):
		return T.maximum(state_below, self.alpha * state_below)
		