import sys
import cv2

class RGBScreenPreprocessor(ScreenPreprocessor):

    def __init__(self, dim=32):
        self.screens = []
        self.dim = dim
        self.channels = 3

    def preprocess(self, screen):
        """
        :description: rescales grayscale screen to be a square of height, width dim
        """
        # currently this just takes the section of the screen with the ball and the block
        height, width, channels = screen.shape
        #screen = screen.reshape(screen.shape[0], screen.shape[1], )
        screen = screen[int(height*.5):, int(width*.05):int(width*.95), :]
        if False:
            cv2.imshow('screen', screen)
            #x = raw_input()
            # # if x == 'x':
            # #     filepath = '/Users/wulfe/Dropbox/School/Stanford/autumn_2015/cs221/project/playing_atari/tests/data/opencv_fe_test_2_img.jpg'
            #       cv2.imwrite(filepath, screen)
        return screen
