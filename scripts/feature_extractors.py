"""
:description: functions for extracting features from screens
"""
import os
import sys
import Queue
import random
import string
import time
import math
import numpy as np
import cv2
import copy
import operator
import itertools
from collections import Counter
from copy import deepcopy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import hashlib
import struct

class CoordinateExtractor(object):

    def __init__(self):
        self.prev_ball_coords = [0,0]
        self.prev_block_coords = [0,0]

    def coordinate_extractor(self, state, action):
        screen = state["screen"]
        screen = screen.reshape(screen.shape[0], screen.shape[1])
        top = screen[:.9 * screen.shape[0],:]
        bot = screen[.9 * screen.shape[0]:,:]
        ball_coords = np.divide(np.unravel_index(top.argmax(), top.shape), map(float, top.shape))
        block_coords = np.divide(np.unravel_index(bot.argmax(), bot.shape), map(float, bot.shape))
        features = [('ball_y', ball_coords[0]),
                    ('ball_x', ball_coords[1]),
                    (('block_x', action), block_coords[1]),
                    ('prev_ball_y', self.prev_ball_coords[0]),
                    ('prev_ball_x', self.prev_ball_coords[1]),
                    (('prev_block_x', action), self.prev_block_coords[1])]
        self.prev_ball_coords = ball_coords
        self.prev_block_coords = block_coords
        return features

    def __call__(self, screen, action):
        return self.coordinate_extractor(screen, action)

class BoundingBoxExtractor(object):

    def print_screen(self, screen):
        screenS = ""
        for x in range(0, screen.shape[0]):
            screenS += "%3d " % x
            for y in range(0, screen.shape[1]):
                screenS += "%3d " % screen.item(x, y, 0)
            screenS += "\n"
        screenS += "   "
        for y in range(0, screen.shape[1]):
            screenS += "%3d " % y
            screenS += "\n"
        print screenS

    def __init__(self):
        self.known_objects = None
        self.object_id_counter = 0
        self.background_color = None
        self.RESERVOIR_SIZE = 10
        self.VERBOSE = False

    def within_tolerance(self, pixel1, pixel2, tolerance_threshold):
        r1 = pixel1 >> 16
        r2 = pixel2 >> 16
        if (abs(r1 - r2) > tolerance_threshold):
            return False
        g1 = (pixel1 >> 8) & 0xFF
        g2 = (pixel2 >> 8) & 0xFF
        if (abs(g1 - g2) > tolerance_threshold):
            return False
        b1 = (pixel1 & 0xFF)
        b2 = (pixel2 & 0xFF)
        if (abs(b1 - b2) > tolerance_threshold):
            return False
        return True

    def extract_object_from_anchor(self, screen, x, y, used):
        tolerance = 40
        explored = 0
        object_val = self.get_rgb_int_val(screen.item(x,y,0), screen.item(x,y,1), screen.item(x,y,2))
        object_coords = []
        min_x = x
        min_y = y
        max_x = x + 1
        max_y = y + 1
        to_explore = Queue.Queue()
        to_explore.put((x, y))
        while not to_explore.empty():
            pos = to_explore.get()
            explored += 1
            pixel_val = self.get_rgb_int_val(screen.item(pos[0],pos[1],0), screen.item(pos[0],pos[1],1), screen.item(pos[0],pos[1],2))
            if pixel_val == self.background_color or \
                    used[pos[0]][pos[1]] or not self.within_tolerance(pixel_val, object_val, tolerance):
                continue
            used[pos[0]][pos[1]] = True
            object_coords.append(pos)
            if pos[0] < min_x:
                min_x = pos[0]
            if pos[1] < min_y:
                min_y = pos[1]
            if pos[0] + 1 > max_x:
                max_x = pos[0] + 1
            if pos[1] + 1 > max_y:
                max_y = pos[1] + 1
            r = (pos[0] + 1, pos[1])
            if (r[0] < screen.shape[0] and not used[r[0]][r[1]]):
                to_explore.put(r)
            l = (pos[0] - 1, pos[1])
            if (l[0] >= 0 and not used[l[0]][l[1]]):
                to_explore.put(l)
            u = (pos[0], pos[1] + 1)
            if (u[1] < screen.shape[1] and not used[u[0]][u[1]]):
                to_explore.put(u)
            d = (pos[0], pos[1] - 1)
            if (d[0] >= 0 and not used[d[0]][d[1]]):
                to_explore.put(d)
        #print "found object with color: " + str(object_val) + " size: " + str(len(object_coords)) + " min: " + str((min_x, min_y)) + " max: " + str((max_x,max_y))
        return (((min_x, min_y), (max_x, max_y)), tuple(object_coords))

    def get_rgb_int_val(self, r, g, b):
        return (r << 16) | (g << 8) | (b)

    def get_bounding_boxes(self, screen):
        if screen.shape[2] != 3:
            print 'rgb screen has depth : ' + str(screen.shape[2])
            return []
        used = np.zeros((screen.shape[0],screen.shape[1]), dtype=bool)
        if self.background_color == None:
            self.background_color = self.identify_background_color(screen)
            print "Found background color: " + str(self.background_color)
        boxes_and_objects = []
        #self.print_screen(screen)
        #print "Searching for objects..."
        for x in range(0, screen.shape[0]):
            for y in range(0, screen.shape[1]):
                screen_color = self.get_rgb_int_val(screen.item(x,y,0),screen.item(x,y,1),screen.item(x,y,2))
                if screen_color == self.background_color:
                    used[x][y] = True
                if not used[x][y]:
                    box_and_object = self.extract_object_from_anchor(screen, x, y, used)
                    boxes_and_objects.append(box_and_object)
        return boxes_and_objects

    def resevoir_sample(self, object_sample, new_sample):
        if len(object_sample) < self.RESERVOIR_SIZE:
            object_sample.append(new_sample)
        else:
            test_sample = random.randint(1,len(object_sample) + 1)
            if test_sample <= len(object_sample):
                object_sample[test_sample - 1] = new_sample

    def is_same_object(self, object_coords, known_object_sample):
        size_test = True #just use object size and set tolerance
        size_test_threshold = 15 #max diff to trigger new object classification
        if (size_test):
            total_len = 0
            for sample in known_object_sample:
                total_len += len(sample)
            avg_len = total_len / len(known_object_sample)
            is_same = (abs(len(object_coords) - avg_len) < size_test_threshold)
            if is_same:
                self.resevoir_sample(known_object_sample, object_coords)
            return is_same
        else:
            return False

    def classify_object(self, box_and_object):
        classified_id = None
        for known_object in self.known_objects:
            if self.is_same_object(box_and_object[1], self.known_objects[known_object]):
                classified_id = known_object
                break
        if classified_id == None:
            new_obj_id = self.object_id_counter
            self.object_id_counter += 1
            print "no match; classifying new object: " + str(new_obj_id)
            self.known_objects[new_obj_id] = [ box_and_object[1] ]
            classified_id = new_obj_id
        return classified_id

    def classify_objects(self, boxes_and_objects):
        classified_boxes_and_objects = []
        if self.known_objects == None:
            self.known_objects = {}
            for box_and_object in boxes_and_objects:
                new_obj_id = self.object_id_counter
                self.object_id_counter += 1
                self.known_objects[new_obj_id] = [ box_and_object[1] ]
                classified_boxes_and_objects.append((new_obj_id, box_and_object[0]))
        else:
            for box_and_object in boxes_and_objects:
                classified_boxes_and_objects.append((self.classify_object(box_and_object), box_and_object[0]))
        return classified_boxes_and_objects


    def identify_background_color(self, screen):
        if screen.shape[2] != 3:
                        print 'rgb screen has depth : ' + str(screen.shape[2])
                        return None
        counts = {}
        max_count = 0
        max_color = None
        for x in range(0, screen.shape[0]):
            for y in range(0, screen.shape[1]):
                pixel_val_r = screen.item(x,y,0)
                pixel_val_g = screen.item(x,y,1)
                pixel_val_b = screen.item(x,y,2)
                pixel_val = self.get_rgb_int_val(pixel_val_r, pixel_val_g, pixel_val_b)
                if pixel_val in counts:
                    counts[pixel_val] += 1
                else:
                    counts[pixel_val] = 1
                if counts[pixel_val] > max_count:
                    max_count = counts[pixel_val]
                    max_color = pixel_val
        if self.VERBOSE:
            print "Found background color: " + str(max_color)
        return max_color

    def __call__(self, state, action):
        screen = state["screen"]
        print_features = False
        if state["objects"] == None:
            millis = int(round(time.time() * 1000))
            boxes_and_objects = self.get_bounding_boxes(screen)
            objects_and_boxes = self.classify_objects(boxes_and_objects)
            state["objects"] = objects_and_boxes
            if self.VERBOSE:
                print "Locating objects took: " + str(int(round(time.time() * 1000)) - millis)
                print "objects: " + str(state["objects"])
                print_features = True
        features = []
        # add singleton features
        for object_and_box in state["objects"]:
            features.append((object_and_box[0], (object_and_box[1])[0][0]))
            features.append((object_and_box[0], (object_and_box[1])[0][1]))
            features.append((object_and_box[0], (object_and_box[1])[1][0]))
            features.append((object_and_box[0], (object_and_box[1])[1][1]))
        # add pairwise features
        for i in range(0, len(state["objects"])):
            object_and_box1 = state["objects"][i]
            for j in range(i + 1, len(state["objects"])):
                object_and_box2 = state["objects"][j]
                pairwise_name = str(i) + "-" + str(j)

                mid_x_1 = ((object_and_box1[1])[0][0] + (object_and_box1[1])[1][0])/2
                mid_x_2 = ((object_and_box2[1])[0][0] + (object_and_box2[1])[1][0])/2
                mid_y_1 = ((object_and_box1[1])[0][1] + (object_and_box1[1])[1][1])/2
                mid_y_2 = ((object_and_box2[1])[0][1] + (object_and_box2[1])[1][1])/2
                diff_mid_x = abs(mid_x_1 - mid_x_2)
                diff_mid_y = abs(mid_y_1 - mid_y_2)
                features.append((pairwise_name + "-xdiff", diff_mid_x))
                features.append((pairwise_name + "-ydiff", diff_mid_y))
        if print_features:
            print "features: " + str(features)
        return features

def get_center(x,y,w,h):
    cx, cy = ((x + w) / 2., (y + h) / 2.)
    return cx, cy

def round_to(value, base):
    return int(base * round(float(value)/base))

class OpenCVBoundingBoxExtractor(object):
	"""
	TODO:
	1. these features need to be cross features
		e.g., (((ball-x:5),(paddle-x:10), (action-left:1)), 1)
	2. need to incorporate action into all these cross features
	3. need to incorporate past position with current position
	"""
	def __init__(self, threshold=10):
		self.iter = 0
		self.found_centers = []
		self.threshold = threshold

	""" is this center inside any previous box? """
	def found_already(self, x, y, w, h):
		cx, cy = get_center(x,y,w,h)
		for fcx, fcy in self.found_centers:
			if math.sqrt((fcx-cx)**2 + (fcy-cy)**2) < self.threshold:
				return True
		self.found_centers.append((cx, cy))
		return False

	def get_bounding_boxes(self, screen):
		self.iter = self.iter + 1;
		img = copy.deepcopy(screen)
		imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(imgray, 20, 100)
		ret, thresh = cv2.threshold(edges, 127, 255, 0)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contours = [cont for cont in contours if cv2.arcLength(cont, True) > 10]

		approx = []
		for cnt in contours:
		    epsilon = 0.00*cv2.arcLength(cnt,True)
		    approx.append(cv2.approxPolyDP(cnt,epsilon,True))

		boxes = []
		for idx, cont in enumerate(approx):
		    x,y,w,h = cv2.boundingRect(cont)
		    get_center(x,y,w,h)
		    if not self.found_already(x,y,w,h):
		    	boxes.append(((x,w),(y,h)))
		#     cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0))
		# cv2.imshow('img_w_boxes', img)
		return boxes

	def __call__(self, state, action):
		screen = state["screen"]
		self.found_centers = []
		try:
			if (len(screen[0][0]) != 3):
				return []
                        state["objects"] = self.get_bounding_boxes(screen)
                except cv2.error as e:
                        # print("encountered exception extracting features, return 0 features")
                        #print(e)
                        return []

		centers = []
		for (x,w), (y,h) in state["objects"]:
			centers.append(get_center(x,y,w,h))
		centers = sorted(centers, key=lambda x: x[1]) # what happens when ball gets below paddle? idx depends on order

		prev_centers = []
		if state["prev_objects"] is not None:
			for (x,w), (y,h) in state["prev_objects"]:
				prev_centers.append(get_center(x,y,w,h))
		prev_centers = sorted(prev_centers, key=lambda x: x[1])

		derivative_pos = []
		if len(centers) == len(prev_centers):
			for c, pc in zip(centers, prev_centers):
				derivative_pos.append(tuple(map(operator.sub, c, pc)))
		return self.get_features_from_centers_derivatives(state, action, centers, derivative_pos)

	def get_features_from_centers_derivatives(self, state, action, centers, derivative_pos):
		features = []
		prev_action_name = 'prev-action-{}'.format(state["prev_action"])
		action_name = 'action-{}'.format(action)
		bucket_size = 10
		pos_names = []
		# base position feature
		for idx, (cx, cy) in enumerate(centers):
			cx = cx/bucket_size
			cy = cy/bucket_size
			name_x = 'object-{}-x-{}'.format(idx, cx)
			name_y = 'object-{}-y-{}'.format(idx, cy)
			pos_names.append(name_x)
			pos_names.append(name_y)
			features.append((name_x, 1))
			features.append((name_y, 1))
			features.append(((action_name, name_x), 1))
			features.append(((action_name, name_y), 1))

		# derivatives
		deriv_names = []
		for idx, (dx, dy) in enumerate(derivative_pos):
			dx = 1 if dx > 0 else -1
			name_x = 'object-{}-dx-{}'.format(idx, dx)
			dy = 1 if dy > 0 else -1
			name_y = 'object-{}-dy-{}'.format(idx, dy)
			deriv_names.append(name_x)
			deriv_names.append(name_y)
			features.append((name_x, 1))
			features.append((name_y, 1))
			features.append(((name_x, action_name), 1))
			# cross with current locations
			diff_bucket_size = 5
			for idx2, (cx, cy) in enumerate(centers):
				cx = cx/bucket_size
				cy = cy/bucket_size
				if idx2 != idx:
					other_x_pos = None
					for idx3, (cx2, cy2) in enumerate(centers):
						cx2 = cx2/bucket_size
						if idx3 == idx:
							other_x_pos = cx2
					name_cross_deriv_pos = 'objects-{}-{}-dx0-{}-dy0-{}-x1-{}'.format(idx,idx2,dx,dy,cx)
					name_cross_deriv_diff = 'objects-{}-{}-dx0-{}-dy0-{}-xdiff-{}'.format(idx,idx2,dx,dy,(cx - other_x_pos)/diff_bucket_size)
					name_cross_deriv_pos_diff = 'objects-{}-{}-x0-{}-dx0-{}-dy0-{}-xdiff-{}'.format(idx,idx2,other_x_pos,dx,dy,(cx - other_x_pos)/diff_bucket_size)
					features.append(((name_cross_deriv_pos, action_name),1))
					features.append(((name_cross_deriv_diff, action_name),1))
					features.append(((name_cross_deriv_pos_diff, action_name),1))
		# differences
		diff_names = []
		for (cx0, cy0), (cx1, cy1) in zip(centers, centers[1:]):
			cx0 = cx0/bucket_size
			cy0 = cy0/bucket_size
			cx1 = cx1/bucket_size
			cy1 = cy1/bucket_size
			diff_x = 'diff-x-pos-{}'.format(cx0 - cx1)
			diff_y = 'diff-y-pos-{}'.format(cy0 - cy1)
			diff_names.append((diff_x, 1))
			diff_names.append((diff_y, 1))
			features.append((diff_x, 1))
			features.append((diff_y, 1))
			features.append(((diff_x, prev_action_name), 1))
			features.append(((diff_x, action_name), 1))
		return features

class IdentityFeatureExtractor(object):
	def __call__(self, state, action):
		return [(k, v) for k, v in state.iteritems()]

class MockBoundingBoxExtractor(OpenCVBoundingBoxExtractor):
	def __init__(self, ball_coords, prev_ball_coords):
		super(MockBoundingBoxExtractor, self).__init__()
		self._prev_ball = prev_ball_coords
		self._ball = ball_coords
		self._paddle_x = 50
		self._paddle_y = 5
		self._paddle_width = 13
		self._paddle_height = 1

	def get_features_paddle_x(self, state, actions, paddle_x):
		all_features = []
		self._paddle_x = paddle_x
		paddle_obj = ((self._paddle_x, self._paddle_y), (self._paddle_x + self._paddle_width, self._paddle_y + self._paddle_height))
		state["prev_objects"] = [paddle_obj, self._prev_ball]
		state["objects"] = [paddle_obj, self._ball]
		state["screen"] = np.zeros((1,1,3))
		for action in actions:
			state["prev_action"] = action
			all_features.append(self(state, action))
		return all_features

class NNetOpenCVBoundingBoxExtractor(object):

    def __init__(self, max_features, threshold=8):
        self.iter = 0
        self.max_features = max_features
        self.threshold = threshold
        self.max_x = 71.5
        self.max_y = 60.

    def get_bounding_boxes(self, screen):
        self.iter = self.iter + 1;
        img = copy.deepcopy(screen)
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(imgray, 20, 100)
        ret, thresh = cv2.threshold(edges, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = [cont for cont in contours if cv2.arcLength(cont, True) > 10]

        approx = []
        for cnt in contours:
            epsilon = 0.00*cv2.arcLength(cnt,True)
            approx.append(cv2.approxPolyDP(cnt,epsilon,True))

        temp_boxes = []
        for idx, cont in enumerate(approx):
            x,y,w,h = cv2.boundingRect(cont)
            temp_boxes.append((x,y,w,h))

        temp_boxes = list(set(temp_boxes))

        if len(temp_boxes) == 0:
            return np.zeros(self.max_features).tolist()
        if len(temp_boxes) == 1:
            return temp_boxes

        temp_boxes = sorted(temp_boxes, key=lambda (x,y,w,h): w*h, reverse=True)
        b1 = temp_boxes[0]
        b2 = temp_boxes[-1]
        cx1, cy1 = get_center(b1[0], b1[1], b1[2], b1[3])
        cx2, cy2 = get_center(b2[0], b2[1], b2[2], b2[3])
        dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        if  dist > self.threshold:
            # cv2.rectangle(img, (b1[0],b1[1]), (b1[0]+b1[2], b1[1]+b1[3]), (0,255,0))
            # cv2.rectangle(img, (b2[0],b2[1]), (b2[0]+b2[2], b2[1]+b2[3]), (0,255,0))
            # cv2.imshow('img_w_boxes', img)
            return [b1, b2]
        else:
            # cv2.rectangle(img, (b1[0],b1[1]), (b1[0]+b1[2], b1[1]+b1[3]), (0,255,0))
            # cv2.imshow('img_w_boxes', img)
            return [b1]

        #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0))


    def __call__(self, state, action):
        screen = state["screen"]
        self.found_centers = []
        try:
            state["objects"] = self.get_bounding_boxes(screen)
        except cv2.error as e:
            # print("encountered exception extracting features, return 0 features")
            print(e)
            return np.zeros(self.max_features)

        centers = []
        for (x,y,w,h) in state["objects"]:
            centers.append((get_center(x,y,w,h), w * h))
        centers = sorted(centers, key=lambda (c, area): area, reverse=True)
        centers = [(cx, cy) for (cx, cy), area in centers]

        prev_centers = []
        if state["prev_objects"] is not None:
            for (x,y,w,h) in state["prev_objects"]:
                prev_centers.append((get_center(x,y,w,h), w * h))
            prev_centers = sorted(prev_centers, key=lambda (c, area): area, reverse=True)
            prev_centers = [(cx, cy) for (cx,cy), area in prev_centers]
        else:
            prev_centers = [(0, 0) * len(centers)]

        position_derivatives = []
        for c, pc in zip(centers, prev_centers):
            dx, dy = c[0] - pc[0], c[1] - pc[1]
            dx = np.clip(dx, -1, 1)
            dy = np.clip(dy, -1, 1)
            position_derivatives.append((dx, dy))

        features = []
        for (cx, cy) in centers:
            features.append(cx / self.max_x)
            features.append(cy / self.max_y)

        while len(features) < self.max_features / 2:
            features.append(0)
        features = features[:4]

        for (dx, dy) in position_derivatives:
            features.append(dx)
            features.append(dy)

        while len(features) < self.max_features:
            features.append(0)
        return features[:self.max_features]


class TrackingClassifyingContourExtractor(object):

    def __init__(self, max_features=100, debug=False):

        #used for subimg storing for debugging
        self.iter = 0
        self.debug = debug
        if self.debug:
            os.system('rm -r subimgs/*')

        #Cap output features CURRENTLY NOT IMPLEMENTED
        self.max_features = max_features

        #Storage of old features for consistent matching
        self.storedFeatures = []
        self.maxStoreSize = 200

        #Ensure you store the previous features
        self.prevFeatures = []

        #The classification algorithm used to cluster features
        #Must have a .fit() method
        #Must have a .labels_ member variable
        self.classifier = DBSCAN(min_samples=1)

        #We need to store some number of examples of each label found so far
        self.MAX_FEATURE_EXAMPLES = 10
        self.featureExamples = {}
        self.numExampleFeatures = 0

        #We need to track how many unique instances of each label we've seen for tracking
        self.labelCounts = Counter()

        #Need to store the previous position and objectId of each feature with a given label
        self.oldFeatureStates = {}


    def __call__(self, state, action):
        screen = state["screen"]
        feats,positions = self.getCurrentFeatures(screen)  #extract raw features about contours from screen
        self.reclassify() #perform clustering on all stored features (includes new features)
        self.consistency() #Ensure labels are consistent from frame to frame

        #Extract current labels
        if self.numExampleFeatures is not 0:
            labels = self.classifier.labels_[-(len(feats)+self.numExampleFeatures):-self.numExampleFeatures]
        else:
            labels = self.classifier.labels_[-(len(feats)):]

        #Add new examples to example tracker
        self.addExamples(feats,labels)

        #Get output features by tracking vs previous frame
        returnFeatures = self.trackFeatures(labels,positions)
        # print returnFeatures
        returnVec = self.createVector(returnFeatures)
        # print labels
        # print returnVec
        prevFeatures = feats
        return returnVec

    def createVector(self,feats):
        vec = []
        for _ in range(0,self.numFeats):
            vec.append(0)
        for feat in feats:
            idxs = []
            cat,id,px,py,dx,dy = feat
            px = px/10
            py = py/10
            dx = dx > 0
            dy = dy > 0
            feat = (cat,px,py,dx,dy)
            test =  struct.pack('f'*len(feat), *feat)

            hash_object = hashlib.md5(test)
            hex_dig = hash_object.hexdigest()
            idx = (int(hex_dig,16))%self.numFeats
            vec[idx] = 1
            idxs.append(idx)
            hash_object = hashlib.sha224(test)
            hex_dig = hash_object.hexdigest()
            idx = (int(hex_dig,16))%self.numFeats
            vec[idx] = 1
            idxs.append(idx)

            hash_object = hashlib.sha256(test)
            hex_dig = hash_object.hexdigest()
            idx = (int(hex_dig,16))%self.numFeats
            vec[idx] = 1
            idxs.append(idx)

            hash_object = hashlib.sha384(test)
            hex_dig = hash_object.hexdigest()
            idx = (int(hex_dig,16))%self.numFeats
            vec[idx] = 1
            idxs.append(idx)
            # print idxs
        return vec

    def getCurrentFeatures(self,screen):
        img = screen #store screen in image
        img2 = deepcopy(img)  #create copy of screen before modifying it
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #create grayscale screen for testing
        edges = cv2.Canny(img, 20, 100)  #perform edge detection

        #The next step closes all detected edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        #detect contours from modified image
        contours = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
        feats = []
        positions = []
        if self.debug:
            self.iter = self.iter + 1
        for idx,cont in enumerate(contours):
            M = cv2.moments(cont)
            x = int(M['m10']/M['m00']) #centroid x
            y = int(M['m01']/M['m00']) #centroid y
            positions.append((x,y))  #add position of feature to structure

            mask = np.zeros(imgray.shape,np.uint8) #create mask
            cv2.drawContours(mask,[cont],0,255,-1) #add contour to mask
            newImg =  cv2.bitwise_and(img2,img2,mask = mask)  #create new image only containing contour
            mean_val = cv2.mean(img2,mask = mask) #get average color of the contour from new image

            #Now we append the feature list that is used for classification
            feats.append( [len(cont),cv2.arcLength(cont,True),mean_val[0],mean_val[1],mean_val[2],cv2.contourArea(cont)])

            #if debugging, output screen captures of each feature at each timestep
            if self.debug:
                outname = 'subimgs/test{}-{}.png'.format(self.iter,idx)
                cv2.imwrite(outname,newImg)

        #finally, store old features for better classification
        for feat in feats:
            self.storedFeatures.append(feat)
        if len(feats)+len(self.prevFeatures) > self.maxStoreSize:
            self.maxStoreSize = len(feats) + len(self.prevFeatures)
        if len(self.storedFeatures) > self.maxStoreSize:
            self.storedFeatures = self.storedFeatures[-self.maxStoreSize:]
        return (feats,positions)

    def reclassify(self):  #Refit to most current data
        feats = deepcopy(self.storedFeatures)
        for key in self.featureExamples:  #Add feature examples to the end so we can find labels later
            for feat in self.featureExamples[key]:
                feats.append(feat)
        feats = np.array(feats)
        feats = StandardScaler().fit_transform(feats)
        self.classifier.fit(feats)
        return

    def consistency(self): #Ensure feautres have constant labels over different timesteps
        newLabels = self.classifier.labels_[-self.numExampleFeatures:]  #Get the labels given in the most recent round
        idx = 0
        relabelings = {}
        for key in self.featureExamples:  #for each previously seen label
            newLabel = newLabels[idx:(idx+len(self.featureExamples[key]))] #find the labels given to the example features
            data = Counter(newLabel)
            newMode = data.most_common(1)[0][0] #find most common new label given
            if newMode is not key: #New iteration assigned different label
                relabelings[newMode] = key
            else: #new iteration assigned same label
                relabelings[key] = key
            idx += len(self.featureExamples[key])
        newLabels = 0
        for idx,oldCat in enumerate(self.classifier.labels_): #for each label assigned in the latest go
            # print relabelings
            if oldCat in relabelings: #if we have a relabeling defined, relabel
                self.classifier.labels_[idx] = relabelings[self.classifier.labels_[idx]]
            else: #Object is new and need a new label accordingly
                self.classifier.labels_[idx] = len(self.featureExamples)+newLabels
                relabelings[oldCat] = len(self.featureExamples)+newLabels
                newLabels +=1
        return

    def addExamples(self,feats,labels):  #add examples if we need new ones
        for idx,feat in enumerate(feats):
            label = labels[idx]
            if label in self.featureExamples:
                if (len(self.featureExamples[label])) < self.MAX_FEATURE_EXAMPLES:
                    self.featureExamples[label].append(feat)
                    self.numExampleFeatures+=1
            else:
                self.featureExamples[label] = [feat]
                self.numExampleFeatures+=1

    #Returns id, dx, dy
    def trackFeatures(self,labels,positions):
        returnFeatures = []
        featureStates = {}
        for idx,label in enumerate(labels): #create dict between a label and all positions that label was found
            if label in featureStates:
                featureStates[label].append(positions[idx])
            else:
                featureStates[label] = [positions[idx]]

        newPids = {}
        for label in featureStates:  #for each label on the current screen
            positions = featureStates[label]
            if label in self.oldFeatureStates:
                oldPids = self.oldFeatureStates[label]
            else:
                oldPids = []
            for oldPos,oldId in oldPids:  #for each position and objectid on the previous screen with the same label
                minDist = float("inf")
                for newPos in positions:  #find the new object with that label that is the closest to the old one
                    dist = math.sqrt((newPos[0]-oldPos[0])**2+(newPos[1]-oldPos[1])**2)
                    if dist < minDist:
                        closest = newPos
                        if dist is 0:
                            break
                        minDist = dist
                positions.remove(closest)  #once found, remove it from the candidate list
                if label in newPids:
                    newPids[label].append((closest,oldId))  #store current position/objectid pair
                else:
                    newPids[label] = [(closest,oldId)]

                returnFeatures.append((label,oldId,closest[0],closest[1],newPos[0]-oldPos[0],newPos[1]-oldPos[1])) #append actual feature
            for pos in positions: #for labeled objects that didnt get matched to the previous frame
                objId = self.labelCounts[label]
                self.labelCounts[label]+=1
                if label in newPids:
                    newPids[label].append((pos,objId))  #store current position/objectid pair
                else:
                    newPids[label] = [(pos,objId)]
                returnFeatures.append(((label,objId,pos[0],pos[1],0,0))) #append actual feature with 0 derivatives since its a new object
        self.oldFeatureStates = newPids
        return returnFeatures
