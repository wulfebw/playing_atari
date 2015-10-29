"""
:description: functions for extracting features from screens
"""
import sys
import Queue
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
		print "found object with color: " + str(object_val) + " size: " + str(len(object_coords)) + " min: " + str((min_x, min_y)) + " max: " + str((max_x,max_y))
		return (((min_x, min_y), (max_x, max_y)), tuple(object_coords))
		

	def get_bounding_boxes(self, screen):
		used = np.zeros(screen.shape, dtype=bool)
		if self.background_color == None:
			self.background_color = self.identify_background_color(screen)
			print "Found background color: " + str(self.background_color)
		boxes_and_objects = []
		#self.print_screen(screen)
		print "Searching for objects..."
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
			print "screen array has dimension: " + str(len(screen.shape))
			return []
		if state["objects"] == None:
			millis = int(round(time.time() * 1000))
			state["objects"] = self.get_bounding_boxes(screen)
			print "Locating objects took: " + str(int(round(time.time() * 1000)) - millis) 
		features = []
		for box, object_coords in state["objects"]:
			features.append((object_coords, box[0][0]))
			features.append((object_coords, box[0][1]))
			features.append((object_coords, box[1][0]))
                        features.append((object_coords, box[1][1]))
		return features
