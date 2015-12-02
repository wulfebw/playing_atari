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
    return ((x + w) / 2, (y + h) / 2)

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
		if len(screen[0][0]) != 3:
			return []

		if state["objects"] == None:
			self.found_centers = []
			state["objects"] = self.get_bounding_boxes(screen)

		centers = []
		for (x,w), (y,h) in state["objects"]:
			centers.append(get_center(x,w,y,h))
		centers = sorted(centers, key=lambda x: x[1]) # what happens when ball gets below paddle? idx depends on order

		prev_centers = []
		if state["prev_objects"] is not None:
			for (x,w), (y,h) in state["prev_objects"]:
				prev_centers.append(get_center(x,w,y,h))
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
		features.append((prev_action_name, 1))
		features.append((action_name, 1))
		features.append(((action_name, prev_action_name), 1))

		bucket_size = 1
		pos_names = []
		# base position feature
		for idx, (cx, cy) in enumerate(centers):
			name_x = 'object-{}-x-{}'.format(idx, cx/bucket_size)
			name_y = 'object-{}-y-{}'.format(idx, cy/bucket_size)
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
			features.append(((name_x, prev_action_name), 1))
			# cross with current locations
			diff_bucket_size = 5
			for idx2, (cx, cy) in enumerate(centers):
				if idx2 != idx:
					other_x_pos = None
					for idx3, (cx2, cy2) in enumerate(centers):
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

    def __init__(self, max_features, threshold=10):
        self.iter = 0
        self.max_features = max_features
        self.found_centers = []
        self.threshold = threshold
        self.max_y = 125.
        self.max_x = 144.

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

    def __call__(self, state):
        screen = state["screen"]
        if screen.shape[-1] != 3:
            print("invalid screen shape: {}".format(screen.shape))
        self.found_centers = []
        try:
            state["objects"] = self.get_bounding_boxes(screen)
        except cv2.error as e:
            # print("encountered exception extracting features, return 0 features")
            print(e)
            return np.zeros(self.max_features)

        centers = []
        for (x,w), (y,h) in state["objects"]:
            centers.append((get_center(x,w,y,h),w * h))
        centers = sorted(centers, key=lambda (c,area): area)
        centers = [c for c, area in centers]

        prev_centers = []
        if state["prev_objects"] is not None:
            for (x,w), (y,h) in state["prev_objects"]:
                prev_centers.append((get_center(x,w,y,h), w * h))
        prev_centers = sorted(prev_centers, key=lambda (c, area): area)
        prev_centers = [c for c, area in prev_centers]

        position_derivatives = []
        if len(centers) == len(prev_centers):
            for c, pc in zip(centers, prev_centers):
                dx, dy = tuple(map(operator.sub, c, pc))
                dx = np.clip(dx, -1, 1)
                dy = np.clip(dy, -1, 1)
                position_derivatives.append((dx, dy))

        features = []
        for (cx, cy), (dx, dy) in zip(centers, position_derivatives):
            features.append(cx / self.max_x)
            features.append(cy / self.max_y)
            features.append(dx)
            features.append(dy)

        while len(features) < self.max_features:
            features.append(0)
        return features[:self.max_features]

class IdentityFeatureExtractor(object):
    def __call__(self, state, action):
        return [(k, v) for k, v in state.iteritems()]
