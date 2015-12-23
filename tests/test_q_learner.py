import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'scripts')))

import collections

import unittest
import numpy as np

import scripts.linear.learning_agents as learning_agents
import scripts.common.feature_extractors as feature_extractors

def get_legal_actions():
	n_actions = 18
	return np.arange(n_actions)

class TestQLearner(unittest.TestCase):
	
	""" incorporateFeedback tests """
	def test_getQ_incorporating_basic_feedback(self):
		legal_actions = get_legal_actions()
		discount = 1
		feature_extractor = feature_extractors.IdentityFeatureExtractor()
		exploration_prob = 0
		step_size = 1
		maxGradient = 1
		num_consecutive_random_actions = 0
		ql = learning_agents.QLearningAlgorithm(legal_actions, discount, feature_extractor,  exploration_prob, step_size, maxGradient, num_consecutive_random_actions)

		state = {'test_feature' : 1}
		new_state = {'test_feature' : 2}
		action = 0
		reward = 1
		ql.incorporateFeedback(state, action, reward, new_state)

		actual = ql.weights
		expected = collections.Counter({'test_feature': 1})
		self.assertEquals(actual, expected)

	def test_getQ_incorporating_multiple_feedback(self):
		legal_actions = get_legal_actions()
		discount = 1
		feature_extractor = feature_extractors.IdentityFeatureExtractor()
		exploration_prob = 0
		step_size = 1
		maxGradient = 1
		num_consecutive_random_actions = 0
		ql = learning_agents.QLearningAlgorithm(legal_actions, discount, feature_extractor,  exploration_prob, step_size, maxGradient, num_consecutive_random_actions)

		state = {'test_feature' : 1}
		new_state = {'test_feature' : 2}
		action = 0
		reward = 1
		ql.incorporateFeedback(state, action, reward, new_state)
		reward = 0
		ql.getAction(new_state)
		ql.incorporateFeedback(state, action, reward, new_state)
		
		actual = ql.weights
		expected = collections.Counter({'test_feature': 2})
		self.assertEquals(actual, expected)



