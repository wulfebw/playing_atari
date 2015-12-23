import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import collections
import cv2

import unittest
import numpy as np

import scripts.common.feature_extractors as feature_extractors

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

class TestNNetOpenCVFeatureExtractor(unittest.TestCase):
    pass
    # """ get_bounding_boxes tests """
    # def test_basic_two_obj(self):

    #     fe = feature_extractors.OpenCVBoundingBoxExtractor()
    #     filepath = os.path.join(DATA_DIR, 'opencv_fe_test_1_img.jpg')
    #     img = cv2.imread(filepath)
    #     actual = fe.get_bounding_boxes(img)
    #     expected = [((107, 21), (109, 6)), ((128, 3), (15, 5))]
    #     self.assertEquals(actual, expected)

    # def test_blur_two_obj(self):

    #     fe = feature_extractors.OpenCVBoundingBoxExtractor()
    #     filepath = os.path.join(DATA_DIR, 'opencv_fe_test_2_img.jpg')
    #     img = cv2.imread(filepath)
    #     actual = fe.get_bounding_boxes(img)
    #     expected = [((47, 20), (109, 6)), ((17, 3), (19, 5))]
    #     self.assertEquals(actual, expected)

    # """ call tests """

    # def test_call_multiple_times_for_objects(self):
    #     filepath_prev = os.path.join(DATA_DIR, 'opencv_fe_test_1_img.jpg')
    #     screen_prev = cv2.imread(filepath_prev)
    #     filepath_cur = os.path.join(DATA_DIR, 'opencv_fe_test_2_img.jpg')
    #     screen_cur = cv2.imread(filepath_cur)

    #     fe = feature_extractors.OpenCVBoundingBoxExtractor()
    #     state = {"screen": screen_prev, "objects": None, "prev_objects": None, "prev_action": 0}
    #     action = 1
    #     fe(state, action)
    #     state = {"screen": screen_cur, "objects": None, "prev_objects": state["objects"], "prev_action": 0}
    #     action = 1
    #     fe(state, action)

    #     actual_objects = state["objects"]
    #     expected_objects = [((47, 20), (109, 6)), ((17, 3), (19, 5))]
    #     self.assertEquals(actual_objects, expected_objects)

    #     actual_prev_objects = state["prev_objects"]
    #     expected_prev_objects = [((107, 21), (109, 6)), ((128, 3), (15, 5))]
    #     self.assertEquals(actual_prev_objects, expected_prev_objects)

    # def test_call_multiple_times_for_features(self):
    #     filepath_prev = os.path.join(DATA_DIR, 'opencv_fe_test_1_img.jpg')
    #     screen_prev = cv2.imread(filepath_prev)
    #     filepath_cur = os.path.join(DATA_DIR, 'opencv_fe_test_2_img.jpg')
    #     screen_cur = cv2.imread(filepath_cur)

    #     fe = feature_extractors.OpenCVBoundingBoxExtractor()
    #     state = {"screen": screen_prev, "objects": None, "prev_objects": None, "prev_action": 0}
    #     action = 1
    #     fe(state, action)
    #     state = {"screen": screen_cur, "objects": None, "prev_objects": state["objects"], "prev_action": 0}
    #     action = 2
    #     actual_features = None
    #     expected_features = None
    #     self.assertEquals(actual_features, expected_features)







