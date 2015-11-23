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

        :type discount: float
        :param discount: the discount value used in the q-learning update

        :type learning_rate: float
        :param learning_rate: the learning rate scales the update value
        """
        self.layers = layers
        self.discount = discount
        self.learning_rate = learning_rate

    def fprop(self, features):
        """
        :description: forward propagates the input through all layers of the network

        :type input: theano.tensor.dvector
        :param input: the original input features to the network as a 1 dimensional col vector

        :type rval outputs[-1]: theano.tensor.dvector
        :param rval outputs[-1]: the output value produced by forward propagating through the enitre network 
                    (i.e., the Q values for each action stored in a vector - so rval[0] would be 
                    the Q value for action 0). 
        """
        outputs = []
        for layer in self.layers:
            features = layer.fprop(features)
            outputs.append(features)
        self.q_values = outputs[-1]
        return outputs[-1]

    def get_loss_and_updates(self, features, action, reward, next_features):
        """
        :description: returns symbolic expressions for the loss and parameter updates

        :type features: theano.tensor.dvector
        :param features: a symbolic vector (1d array) representing the current features 

        :type action: theano.tensor.lscalar
        :param action: the action previously taken by the agent. used to index into the output values

        :type reward: theano.tensor.dscalar
        :param reward: the reward value returned by the ALE

        :type next_features: theano.tensor.dvector
        :param next_features: symbolic vector representing the features from the next state

        :type rval loss: theano.tensor.dscalar
        :param rval loss: the loss resulting from this current q-learning update

        :type rval updates: list of 2-tuples 
        :param rval updates: tuples mapping a parameter to its update resulting from this iteration
        """
        q_values = self.fprop(features)
        next_q_values = self.fprop(next_features)
        target = reward + self.discount * T.max(next_q_values)
        loss = .5 * T.sqr(target - q_values[action])

        params = self.get_params()
        gparams = T.grad(loss, params)
        updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(params, gparams)]

        return (loss, updates)

    def get_params(self):
        """
        :description: return a list of the parameters from each layer of this network (1-d list)
        """
        params = []
        for layer in self.layers:
            params += layer.params
        return params

class HiddenLayer(object):
    """
    :description: generic network hidden layer.
    """

    def __init__(self, n_vis, n_hid, layer_name, param_init_range=0.02):
        """
        :type n_vis: int 
        :param n_vis: the number of input nodes to this layer (i.e., input space)

        :type n_hid: int
        :param n_hid: the number of output nodes of this layer (i.e., output space)
        
        :type layer_name: string
        :param layer_name: the name used in labeling the parameters of this layer

        :type param_init_range: float
        :param param_init_range: the magnitude of values in which weights should be initialized
        """
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.layer_name = layer_name
        self.rng = np.random.RandomState()
        self.param_init_range = param_init_range
        self.activation = theano.tensor.nnet.sigmoid

        init_W = self.rng.uniform(-self.param_init_range, self.param_init_range, (self.n_vis, self.n_hid))
        self.W = theano.shared(value=init_W, name=self.layer_name + '_W', borrow=True)
        self.b = theano.shared(value=np.zeros(self.n_hid), name=self.layer_name + '_b', borrow=True)

        self.params = [self.W, self.b]

    def fprop(self, state_below):
        """
        :description: forward propagate the state from the layer below this layer. This involves
                    taking the dot product between a row vector of dimensions (1 x n_vis) and the weight 
                    matrix of this current layer of dimensions (n_vis x n_hid).

        :type state_below: theano.tensor.dvector
        :param state_below: state_below is the state from the layer below this one. 
                            This will originally be the feature vector. 
        """
        return self.activation(T.dot(state_below, self.W) + self.b)

class OutputLayer(object):
    """
    :description: an output layer class that performs a linear operation on the output of the layer
                  below this one.
    """

    def __init__(self, layer_name, alpha=0.5):
        """
        :type layer_name: string
        :param layer_name: the name used in labeling the parameters of this layer

        :type alpha: float
        :param alpha: this layer currently acts as a rectified linear unit and alpha is 
                        the parameter of this function which determines how negative the 
                        layer allows its output to be. 0 <= alpha <= 1
        """
        self.layer_name = layer_name
        self.alpha = alpha
        self.params = []

    def fprop(self, state_below):
        """
        :description: forward propagate the state below through this layer. Specifically
                        acts as a rectified linear unit, which takes the elementwise max
                        of the state_below and some scaling of the state_below. 
        """
        return T.maximum(state_below, self.alpha * state_below)
        