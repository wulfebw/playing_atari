import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import collections
import cv2

import unittest
import numpy as np

import scripts.linear.eligibility_traces as eligibility_traces

class TestEligibilityTraces(unittest.TestCase):

    """ __setitem__ tests """
    def test_setitem_exists(self):
        etraces = eligibility_traces.EligibilityTraces(threshold=.5, decay=.5)
        etraces['a'] = 1
        predicate = 'a' in etraces
        self.assertTrue(predicate)

    def test_setitem_clip(self):
        etraces = eligibility_traces.EligibilityTraces(threshold=.5, decay=.5)
        etraces['a'] = 1.5
        predicate = 'a' in etraces
        self.assertTrue(predicate)
        actual = etraces['a']
        expected = 1
        self.assertEquals(actual, expected)

    """ update_all tests """
    def test_decay_removal(self):
        etraces = eligibility_traces.EligibilityTraces(threshold=.5, decay=.49)
        etraces['a'] = 1
        etraces.update_all()
        predicate = 'a' in etraces
        self.assertTrue(predicate)
        etraces.update_all()
        predicate = 'a' in etraces
        self.assertFalse(predicate)

    """ iteration tests """
    def test_iteration(self):
        etraces = eligibility_traces.EligibilityTraces(threshold=.5, decay=.5)
        etraces['a'] = 1
        etraces['b'] = 2
        actual = []
        for k, v in etraces.iteritems():
            actual.append(k)
        expected = ['a', 'b']
        self.assertEquals(actual, expected)


if __name__ == '__main__':
    # this runs all tests in file if executed directly (e.g., python test_nnet.py)
    # run a single test by specifying the name (e.g., python test_nnet.py 
    # TestNNet.test_loss_updates_one_layer_positive_relu)
    unittest.main()
