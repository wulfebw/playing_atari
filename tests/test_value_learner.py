import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'scripts')))

import collections

import unittest
import numpy as np

import learning_agents
import feature_extractors

def get_legal_actions():
    n_actions = 18
    return np.arange(n_actions)

class TestValueLearner(unittest.TestCase):
    
    """ getQ tests """
    def test_getQ_init(self):
        legal_actions = get_legal_actions()
        discount = 1
        feature_extractor = feature_extractors.IdentityFeatureExtractor()
        exploration_prob = .3
        ql = learning_agents.ValueLearningAlgorithm(legal_actions, discount, feature_extractor,  exploration_prob)

        state = {}
        action = 0
        actual = ql.getQ(state, action)
        expected = 0
        self.assertEquals(actual, expected)