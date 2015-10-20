"""
:description: functions for extracting features from screens
"""
import sys

import numpy as np

class CoordinateExtractor(object):

	def __init__(self):
		self.prev_ball_coords = [0,0]
		self.prev_block_coords = [0,0]

	def coordinate_extractor(self, screen, action):
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


