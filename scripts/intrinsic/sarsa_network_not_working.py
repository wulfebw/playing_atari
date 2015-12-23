"""
SARSA(Lambda) with CNN function approximator
"""
import lasagne
import numpy as np
import theano
import theano.tensor as T
from updates import deepmind_rmsprop

from lasagne.layers import cuda_convnet


class DeepSARSALearner:
    """
    Deep SARSA(Lambda) network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, network_type, update_rule, 
                 lambda_decay, rng, input_scale=255.0):

        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.discount = discount
        self.lr = learning_rate
        self.rho = rho
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.lambda_decay = lambda_decay
        self.rng = rng
        lasagne.random.set_rng(self.rng)
        self.update_counter = 0

        self.l_out = self.build_network(network_type, input_width, input_height, num_actions, num_frames)

        state = T.tensor3('state')
        action = T.iscalar('action')
        reward = T.dscalar('reward')
        next_state = T.tensor3('next_state')
        next_action = T.iscalar('next_action')
        
        # S A R S' A'
        self.state_shared = theano.shared(
            np.zeros((num_frames, input_height, input_width),
                     dtype=theano.config.floatX))
        self.action_shared = theano.shared(0, dtype='int32'))
        self.reward_shared = theano.shared(0, dtype=theano.config.floatX))
        self.next_state_shared = theano.shared(
            np.zeros((num_frames, input_height, input_width),
                     dtype=theano.config.floatX))
        self.next_action_shared = theano.shared(0, dtype='int32'))
        
        # create symbolic graph of loss
        q_vals = lasagne.layers.get_output(self.l_out, state / input_scale)
        next_q_vals = lasagne.layers.get_output(self.l_out, next_state / input_scale)
        next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        target = reward + self.discount * next_q_val[next_action]
        diff = target - q_vals[action]
        loss = 0.5 * diff ** 2

        params = lasagne.layers.helper.get_all_params(self.l_out)  
        givens = {
            state: self.state_shared,
            action: self.action_shared,
            reward: self.reward_shared,
            next_state: self.next_state_shared,
            next_action: self.next_action_shared
        }

        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho, self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho, self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None, self.momentum)

        self._train = theano.function([], [loss, q_vals], updates=updates, givens=givens)
        self._q_vals = theano.function([], q_vals, givens={state: self.state_shared})

    def build_network(self, network_type, input_width, input_height, output_dim, num_frames):
        if network_type == "large":
            return self.build_large_network(input_width, input_height, output_dim, num_frames)
        elif network_type == "small":
            return self.build_small_network(input_width, input_height, output_dim, num_frames)
        elif network_type == "linear":
            return self.build_linear_network(input_width, input_height, output_dim, num_frames)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))

    def train(self, state, action, reward, next_state, next_action):
        """
        Train one sarsa tuple.

        Arguments:
        note: f = num frames, h = height, w = width
        state - f x h x w numpy array
        action - int
        reward - double
        next_state - f x h x w numpy array
        
        Returns: average loss
        """
        self.state_shared.set_value(state)
        self.action_shared.set_value(action)
        self.reward_shared.set_value(reward)
        self.next_state_shared.set_value(next_state)
        self.next_action_shared.set_value(next_action)

        loss, _ = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state):
        self.state_shared.set_value(state)
        return self._q_vals()

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def build_large_network(self, input_width, input_height, output_dim, num_frames):
        l_in = lasagne.layers.InputLayer(
            shape=(num_frames, input_width, input_height)
        )

        l_conv1 = cuda_convnet.Conv2DCCLayer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv3 = cuda_convnet.Conv2DCCLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_small_network(self, input_width, input_height, output_dim, num_frames):
        l_in = lasagne.layers.InputLayer(
            shape=(num_frames, input_width, input_height)
        )

        l_conv1 = cuda_convnet.Conv2DCCLayer(
            l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_linear_network(self, input_width, input_height, output_dim, num_frames):
        l_in = lasagne.layers.InputLayer(
            shape=(num_frames, input_width, input_height)
        )

        l_out = lasagne.layers.DenseLayer(
            l_in,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.Constant(0.0),
            b=None
        )

        return l_out

if __name__ == '__main__':
    pass