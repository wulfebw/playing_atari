import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'scripts')))

import collections

import unittest
import numpy as np
import theano
import theano.tensor as T

from mlp import MLP, HiddenLayer, OutputLayer


class TestNNet(unittest.TestCase):
    
    """ fprop tests """
    def test_fprop_single_layer_zero_weights_positive_input_values_relu(self):
        hidden_layer = HiddenLayer(n_vis=4, n_hid=2, layer_name='h', activation='relu', param_init_range=0, alpha=0)
        mlp = MLP([hidden_layer], discount=1, learning_rate=1)
        features = [1, 2, 3, 4]
        actual = list(mlp.fprop(features).eval())
        expected = [0., 0.]
        self.assertSequenceEqual(actual, expected)

    def test_fprop_single_layer_one_weights_positive_input_values_relu(self):
        n_vis = 4
        n_hid = 2
        hidden_layer = HiddenLayer(n_vis=n_vis, n_hid=n_hid, layer_name='h', activation='relu', param_init_range=0, alpha=0)
        W = theano.shared(value=np.ones((n_vis, n_hid)), name='h_W', borrow=True)
        hidden_layer.W = W
        mlp = MLP([hidden_layer], discount=1, learning_rate=1)
        features = [1, 2, 3, 4]
        actual = list(mlp.fprop(features).eval())
        expected = [10., 10.]
        self.assertSequenceEqual(actual, expected)

    def test_fprop_single_layer_one_weights_negative_input_values_relu(self):
        n_vis = 4
        n_hid = 2
        hidden_layer = HiddenLayer(n_vis=n_vis, n_hid=n_hid, layer_name='h', activation='relu', param_init_range=0, alpha=0)
        W = theano.shared(value=np.ones((n_vis, n_hid)), name='h_W', borrow=True)
        hidden_layer.W = W
        mlp = MLP([hidden_layer], discount=1, learning_rate=1)
        features = [-1, 2, -3, 4]
        actual = list(mlp.fprop(features).eval())
        expected = [2., 2.]
        self.assertSequenceEqual(actual, expected)

    def test_fprop_single_layer_one_weights_negative_output_values_relu(self):
        n_vis = 4
        n_hid = 2
        hidden_layer = HiddenLayer(n_vis=n_vis, n_hid=n_hid, layer_name='h', activation='relu', param_init_range=0, alpha=0)
        W = theano.shared(value=np.ones((n_vis, n_hid)), name='h_W', borrow=True)
        hidden_layer.W = W
        mlp = MLP([hidden_layer], discount=1, learning_rate=1)
        features = [-1, -2, -3, 4]
        actual = list(mlp.fprop(features).eval())
        expected = [0., 0.]
        self.assertSequenceEqual(actual, expected)

    def test_fprop_double_layer_one_weights_positive_output_values_relu(self):
        n_vis = 8
        n_hid = 2
        hidden_layer_1 = HiddenLayer(n_vis=n_vis, n_hid=n_vis / 2, layer_name='h1', activation='relu', param_init_range=0, alpha=0)
        hidden_layer_2 = HiddenLayer(n_vis=n_vis / 2, n_hid=n_hid, layer_name='h2', activation='relu', param_init_range=0, alpha=0)
        W = theano.shared(value=np.ones((n_vis, n_vis / 2)), name='h1_W', borrow=True)
        hidden_layer_1.W = W
        W = theano.shared(value=np.ones((n_vis / 2, n_hid)), name='h2_W', borrow=True)
        hidden_layer_2.W = W
        mlp = MLP([hidden_layer_1, hidden_layer_2], discount=1, learning_rate=1)
        features = np.ones(n_vis)
        actual = list(mlp.fprop(features).eval())
        expected = [32., 32.]
        self.assertSequenceEqual(actual, expected)

    def test_fprop_double_layer_one_weights_negative_output_values_relu(self):
        n_vis = 8
        n_hid = 2
        hidden_layer_1 = HiddenLayer(n_vis=n_vis, n_hid=n_vis / 2, layer_name='h1', activation='relu', param_init_range=0, alpha=0)
        hidden_layer_2 = HiddenLayer(n_vis=n_vis / 2, n_hid=n_hid, layer_name='h2', activation='relu', param_init_range=0, alpha=0)
        W = theano.shared(value=np.ones((n_vis, n_vis / 2)), name='h1_W', borrow=True)
        hidden_layer_1.W = W
        W = theano.shared(value=np.ones((n_vis / 2, n_hid)), name='h2_W', borrow=True)
        hidden_layer_2.W = W
        mlp = MLP([hidden_layer_1, hidden_layer_2], discount=1, learning_rate=1)
        features = [-5, -4, -3, -2, -1, 0, 1, 2]
        actual = list(mlp.fprop(features).eval())
        expected = [0., 0.]
        self.assertSequenceEqual(actual, expected)

    """ get_loss_and_updates tests """
    def test_loss_updates_one_layer_positive_relu(self):
        n_vis = 4
        n_hid = 2
        hidden_layer = HiddenLayer(n_vis=n_vis, n_hid=n_hid, layer_name='h', activation='relu', param_init_range=0, alpha=0)
        # W = theano.shared(value=np.ones((n_vis, n_hid)), name='h_W', borrow=True)
        # hidden_layer.W = W
        mlp = MLP([hidden_layer], discount=1, learning_rate=1)
        
        features = T.dvector('features')
        action = T.lscalar('action')
        reward = T.dscalar('reward')
        next_features = T.dvector('next_features')
        loss, updates = mlp.get_loss_and_updates(features, action, reward, next_features)
        train = theano.function(
                    [features, action, reward, next_features],
                    outputs=loss,
                    updates=updates,
                    mode='FAST_COMPILE')

        features = [1,1,1,1]
        action = 0
        reward = 1
        next_features = [1,1,1,1]

        actual_loss = train(features, action, reward, next_features)
        expected_loss = 0.5

        actual_weights = list(mlp.layers[0].W.eval())
        expected_weights = [[1,0], [1,0], [1,0], [1,0]]

        self.assertEqual(actual_loss, expected_loss)
        self.assertTrue(np.array_equal(actual_weights, expected_weights))

    def test_loss_updates_one_layer_positive_diff_action_relu(self):
        n_vis = 4
        n_hid = 2
        hidden_layer = HiddenLayer(n_vis=n_vis, n_hid=n_hid, layer_name='h', activation='relu', param_init_range=0, alpha=0)
        # W = theano.shared(value=np.ones((n_vis, n_hid)), name='h_W', borrow=True)
        # hidden_layer.W = W
        mlp = MLP([hidden_layer], discount=1, learning_rate=1)
        
        features = T.dvector('features')
        action = T.lscalar('action')
        reward = T.dscalar('reward')
        next_features = T.dvector('next_features')
        loss, updates = mlp.get_loss_and_updates(features, action, reward, next_features)
        train = theano.function(
                    [features, action, reward, next_features],
                    outputs=loss,
                    updates=updates,
                    mode='FAST_COMPILE')

        features = [1,1,1,1]
        action = 1
        reward = 1
        next_features = [1,1,1,1]

        actual_loss = train(features, action, reward, next_features)
        expected_loss = 0.5

        actual_weights = list(mlp.layers[0].W.eval())
        expected_weights = [[0,1], [0,1], [0,1], [0,1]]

        self.assertEqual(actual_loss, expected_loss)
        self.assertTrue(np.array_equal(actual_weights, expected_weights))

    def test_loss_updates_one_layer_negative_features_relu(self):
        n_vis = 4
        n_hid = 2
        hidden_layer = HiddenLayer(n_vis=n_vis, n_hid=n_hid, layer_name='h', activation='relu', param_init_range=0, alpha=0)
        # W = theano.shared(value=np.ones((n_vis, n_hid)), name='h_W', borrow=True)
        # hidden_layer.W = W
        mlp = MLP([hidden_layer], discount=1, learning_rate=1)
        
        features = T.dvector('features')
        action = T.lscalar('action')
        reward = T.dscalar('reward')
        next_features = T.dvector('next_features')
        loss, updates = mlp.get_loss_and_updates(features, action, reward, next_features)
        train = theano.function(
                    [features, action, reward, next_features],
                    outputs=loss,
                    updates=updates,
                    mode='FAST_COMPILE')

        features = [-1,-1,-1,-1]
        action = 0
        reward = 1
        next_features = [1,1,1,1]

        actual_loss = train(features, action, reward, next_features)
        expected_loss = 0.5

        actual_weights = mlp.layers[0].W.eval().tolist()
        expected_weights = [[-1,0], [-1,0], [-1,0], [-1,0]]

        self.assertEqual(actual_loss, expected_loss)
        self.assertSequenceEqual(actual_weights, expected_weights)

    def test_loss_updates_one_layer_positive_features_with_positive_weights_relu(self):
        n_vis = 4
        n_hid = 2
        hidden_layer = HiddenLayer(n_vis=n_vis, n_hid=n_hid, layer_name='h', activation='relu', param_init_range=0, alpha=0)
        hidden_layer.W.set_value(np.ones((n_vis, n_hid)))
        mlp = MLP([hidden_layer], discount=1, learning_rate=1)
        
        features = T.dvector('features')
        action = T.lscalar('action')
        reward = T.dscalar('reward')
        next_features = T.dvector('next_features')
        loss, updates = mlp.get_loss_and_updates(features, action, reward, next_features)
        train = theano.function(
                    [features, action, reward, next_features],
                    outputs=loss,
                    updates=updates,
                    mode='FAST_COMPILE')

        features = [1,1,1,1]
        action = 0
        reward = 1
        next_features = [1,1,1,1]

        actual_loss = train(features, action, reward, next_features)
        expected_loss = 0.5

        actual_weights = mlp.layers[0].W.eval().tolist()
        expected_weights = [[2,1], [2,1], [2,1], [2,1]]

        self.assertEqual(actual_loss, expected_loss)
        self.assertSequenceEqual(actual_weights, expected_weights)

    def test_loss_updates_one_layer_positive_features_with_negative_weights_relu(self):
        n_vis = 4
        n_hid = 2
        hidden_layer = HiddenLayer(n_vis=n_vis, n_hid=n_hid, layer_name='h', activation='relu', param_init_range=0, alpha=0)
        hidden_layer.W.set_value(np.ones((n_vis, n_hid)) * -1)
        mlp = MLP([hidden_layer], discount=1, learning_rate=1)
        
        features = T.dvector('features')
        action = T.lscalar('action')
        reward = T.dscalar('reward')
        next_features = T.dvector('next_features')
        loss, updates = mlp.get_loss_and_updates(features, action, reward, next_features)
        train = theano.function(
                    [features, action, reward, next_features],
                    outputs=loss,
                    updates=updates,
                    mode='FAST_COMPILE')

        features = [1,1,1,1]
        action = 0
        reward = 1
        next_features = [1,1,1,1]

        actual_loss = train(features, action, reward, next_features)
        expected_loss = 0.5

        actual_weights = mlp.layers[0].W.eval().tolist()
        expected_weights = [[-1,-1], [-1,-1], [-1,-1], [-1,-1]]

        self.assertEqual(actual_loss, expected_loss)
        self.assertSequenceEqual(actual_weights, expected_weights)

if __name__ == '__main__':
    # this runs all tests in file if executed directly (e.g., python test_nnet.py)
    # run a single test by specifying the name (e.g., python test_nnet.py 
    # TestNNet.test_loss_updates_one_layer_positive_relu)
    unittest.main()