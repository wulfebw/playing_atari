"""
:description: functions for extracting features from screens
"""
import sys
import Queue
import random
import string
import time
import numpy as np

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
		print "found object with color: " + str(object_val) + " size: " + str(len(object_coords)) + " min: " + str((min_x, min_y)) + " max: " + str((max_x,max_y))
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
		print "Searching for objects..."
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
