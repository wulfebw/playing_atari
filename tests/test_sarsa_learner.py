import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'scripts')))

import collections

import unittest
import numpy as np

import scripts.linear.learning_agents as learning_agents
import scripts.common.feature_extractors as feature_extractors

def get_legal_actions():
    n_actions = 4
    return np.arange(n_actions)

class TestSARSALearner(unittest.TestCase):
    
    """ incorporateFeedback tests """
    def test_getQ_incorporating_basic_feedback(self):
        legal_actions = get_legal_actions()
        discount = 1
        feature_extractor = feature_extractors.IdentityFeatureExtractor()
        exploration_prob = 0
        step_size = 1
        maxGradient = 5
        num_consecutive_random_actions = 0
        ql = learning_agents.SARSALearningAlgorithm(legal_actions, discount, feature_extractor,  exploration_prob, step_size, maxGradient, num_consecutive_random_actions)

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
        maxGradient = 5
        num_consecutive_random_actions = 0
        ql = learning_agents.SARSALearningAlgorithm(legal_actions, discount, feature_extractor,  exploration_prob, step_size, maxGradient, num_consecutive_random_actions)

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

class TestSARSALambdaLearner(unittest.TestCase):
    
    """ incorporateFeedback tests """
    def test_incorporating_feedback_multiupdate_with_lambda_decay(self):
        legal_actions = get_legal_actions()
        discount = 1
        feature_extractor = feature_extractors.IdentityFeatureExtractor()
        exploration_prob = 0
        step_size = 1
        decay = .5
        maxGradient = 10
        threshold = .001
        maxGradient = 5
        num_consecutive_random_actions = 0
        ql = learning_agents.SARSALambdaLearningAlgorithm(legal_actions, discount, 
            feature_extractor, exploration_prob, step_size, threshold, decay, maxGradient,num_consecutive_random_actions)

        state_1 = {'test_feature_1' : 1}
        state_2 = {'test_feature_2' : 1}
        state_3 = {'test_feature_3' : 1}
        state_4 = {'test_feature_4' : 1}
        action = 0
        reward = 1
        ql.incorporateFeedback(state_1, action, reward, state_2)
        ql.incorporateFeedback(state_2, action, reward, state_3)
        ql.incorporateFeedback(state_3, action, reward, state_4)

        
        actual = ql.weights
        expected = collections.Counter({'test_feature_1': 1.75, 
                    'test_feature_2': 1.5, 'test_feature_3': 1})
        self.assertEquals(actual, expected)

    def test_incorporating_feedback_multiupdate_with_lambda_decay_diff_value(self):
        legal_actions = get_legal_actions()
        discount = 1
        feature_extractor = feature_extractors.IdentityFeatureExtractor()
        exploration_prob = 0
        step_size = 1
        decay = .2
        maxGradient = 10
        threshold = .001
        maxGradient = 5
        num_consecutive_random_actions = 0
        ql = learning_agents.SARSALambdaLearningAlgorithm(legal_actions, discount, 
            feature_extractor, exploration_prob, step_size, threshold, decay, maxGradient,
            num_consecutive_random_actions)

        state_1 = {'test_feature_1' : 1}
        state_2 = {'test_feature_2' : 1}
        state_3 = {'test_feature_3' : 1}
        state_4 = {'test_feature_4' : 1}
        action = 0
        reward = 1
        ql.incorporateFeedback(state_1, action, reward, state_2)
        ql.incorporateFeedback(state_2, action, reward, state_3)
        ql.incorporateFeedback(state_3, action, reward, state_4)

        
        actual = ql.weights
        expected = collections.Counter({'test_feature_1': 1.24, 
                    'test_feature_2': 1.2, 'test_feature_3': 1})
        self.assertEquals(actual, expected)

    def test_incorporating_feedback_overlapping_multiupdate_with_decay(self):
        legal_actions = get_legal_actions()
        discount = 1
        feature_extractor = feature_extractors.IdentityFeatureExtractor()
        exploration_prob = 0
        step_size = 1
        decay = .5
        maxGradient = 10
        threshold = 0
        maxGradient = 5
        num_consecutive_random_actions = 0
        ql = learning_agents.SARSALambdaLearningAlgorithm(legal_actions, discount, 
            feature_extractor, exploration_prob, step_size, threshold, decay, maxGradient,
            num_consecutive_random_actions)

        state_1 = {'test_feature_1' : 1}
        state_2 = {'test_feature_2' : 1}
        state_3 = {'test_feature_1' : 1}
        action = 0
        reward = 1
        ql.incorporateFeedback(state_1, action, reward, state_2)
        ql.incorporateFeedback(state_2, action, reward, state_3)

        
        actual = ql.weights
        expected = collections.Counter({'test_feature_1': 2, 'test_feature_2': 2})
        self.assertEquals(actual, expected)

    def test_incorporating_feedback_overlapping_multiupdate_with_decay_negative(self):
        legal_actions = get_legal_actions()
        discount = 1
        feature_extractor = feature_extractors.IdentityFeatureExtractor()
        exploration_prob = 0
        step_size = 1
        decay = .5
        maxGradient = 10
        threshold = .001
        maxGradient = 5
        num_consecutive_random_actions = 0
        ql = learning_agents.SARSALambdaLearningAlgorithm(legal_actions, discount, 
            feature_extractor, exploration_prob, step_size, threshold, decay, maxGradient, 
            num_consecutive_random_actions)

        state_1 = {'test_feature_1' : 1}
        # ql.weights['test_feature_2'] = 1
        state_2 = {'test_feature_2' : -1}
        state_3 = {'test_feature_1' : 1}
        action = 0
        reward_1 = 1
        reward_2 = -1
        ql.incorporateFeedback(state_1, action, reward_1, state_2)
        ql.incorporateFeedback(state_2, action, reward_2, state_3)

        
        actual = ql.weights
        expected = collections.Counter({'test_feature_1': 1, 
                    'test_feature_2': 0})
        self.assertEquals(actual, expected)

    def test_incorporating_feedback_overlapping_multiupdate_with_decay_more_than_one(self):
        legal_actions = get_legal_actions()
        discount = 1
        feature_extractor = feature_extractors.IdentityFeatureExtractor()
        exploration_prob = 0
        step_size = 1
        decay = .5
        maxGradient = 10
        threshold = .001
        maxGradient = 5
        num_consecutive_random_actions = 0
        ql = learning_agents.SARSALambdaLearningAlgorithm(legal_actions, discount, 
            feature_extractor, exploration_prob, step_size, threshold, decay, maxGradient, 
            num_consecutive_random_actions)

        state_1 = {'test_feature_1' : 1}
        ql.weights['test_feature_2'] = 1
        state_2 = {'test_feature_2' : 2}
        state_3 = {'test_feature_1' : 1}
        action = 0
        reward = 1
        ql.incorporateFeedback(state_1, action, reward, state_2)
        ql.incorporateFeedback(state_2, action, reward, state_3)

        
        actual = ql.weights
        expected = collections.Counter({'test_feature_1': 4, 
                    'test_feature_2': 3})
        self.assertEquals(actual, expected)

if __name__ == '__main__':
    # this runs all tests in file if executed directly (e.g., python test_nnet.py)
    # run a single test by specifying the name (e.g., python test_nnet.py 
    # TestNNet.test_loss_updates_one_layer_positive_relu)
    unittest.main()