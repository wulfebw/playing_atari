"""
:description: functions for extracting features from screens
"""
import os
import sys
import Queue
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
				screenS += "%3d " % screen.item(x, y)
			screenS += "\n"
		screenS += "   "
		for y in range(0, screen.shape[1]):
                        screenS += "%3d " % y
                screenS += "\n"
		print screenS
				
	def __init__(self):
		self.known_objects = {}
		self.object_id = 0
		self.background_color = None

	def extract_object_from_anchor(self, screen, x, y, used):
		tolerance = 40
		explored = 0
		object_val = screen[x][y]
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
			if screen.item(pos[0],pos[1]) == self.background_color or \
					used[pos[0]][pos[1]] or abs(screen.item(pos[0],pos[1]) - object_val) > tolerance:
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
		

	def get_bounding_boxes(self, screen):
		used = np.zeros(screen.shape, dtype=bool)
		if self.background_color == None:
			self.background_color = self.identify_background_color(screen)
			print "Found background color: " + str(self.background_color)
		boxes_and_objects = []
		#self.print_screen(screen)
		#print "Searching for objects..."
		for x in range(0, screen.shape[0]):
                        for y in range(0, screen.shape[1]):
				if screen[x][y] == self.background_color:
					used[x][y] = True
				if not used[x][y]:
					box_and_object = self.extract_object_from_anchor(screen, x, y, used)
					boxes_and_objects.append(box_and_object)
		return boxes_and_objects

	def identify_background_color(self, screen):
		counts = {}
		max_count = 0
		max_color = None
		for x in range(0, screen.shape[0]):
			for y in range(0, screen.shape[1]):
				pixel_val = screen.item(x,y)
				print pixel_val
				if pixel_val in counts:
					counts[pixel_val] += 1
				else:
					counts[pixel_val] = 1
				if counts[pixel_val] > max_count:
					max_count = counts[pixel_val]
					max_color = pixel_val
		return max_color

	def __call__(self, state, action):
		screen = state["screen"]
		if len(screen.shape) != 2:
			#print "screen array has dimension: " + str(len(screen.shape))
			return []
		if state["objects"] == None:
			millis = int(round(time.time() * 1000))
			state["objects"] = self.get_bounding_boxes(screen)
			#print "Locating objects took: " + str(int(round(time.time() * 1000)) - millis) 
		features = []
		for box, object_coords in state["objects"]:
			print object_coords
			sys.exit(0)
			features.append((object_coords, box[0][0]))
			features.append((object_coords, box[0][1]))
			features.append((object_coords, box[1][0]))
			features.append((object_coords, box[1][1]))
		features.append(('action', action))
		for thing in features:
			print thing
			print '\n'
		sys.exit(0)
		return features


def get_center(x,y,w,h):
	return ((x + w) / 2, (y + h) / 2)

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
		centers = sorted(centers, key=lambda x: x[1])

		prev_centers = []
		if state["prev_objects"] is not None:
			for (x,w), (y,h) in state["prev_objects"]:
				prev_centers.append(get_center(x,w,y,h))
		prev_centers = sorted(prev_centers, key=lambda x: x[1])

		derivative_pos = []
		if len(centers) == len(prev_centers):
			for c, pc in zip(centers, prev_centers):
				derivative_pos.append(tuple(map(operator.sub, c, pc)))

		features = []
		prev_action_name = 'prev-action-{}'.format(state["prev_action"])
		action_name = 'action-{}'.format(action)
		features.append((prev_action_name, 1))
		features.append((action_name, 1))
		features.append(((action_name, prev_action_name), 1))

		pos_names = []
		# base position feature
		for idx, (cx, cy) in enumerate(centers):
			name_x = 'object-{}-x-{}'.format(idx, round(cx, -1))
			name_y = 'object-{}-y-{}'.format(idx, round(cy, -1))
			pos_names.append(name_x)
			pos_names.append(name_y)
			features.append((name_x, 1))
			features.append((name_y, 1))
			features.append(((action_name, name_x), 1))

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
