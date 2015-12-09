"""
NOTE: Parts of this file are adapted from the Stanford class CS221 Artificial Intelligence.
      It is from a homework assignement on reinforcement learning.
"""

import sys, collections, math, random
import numpy as np

MAX_FEATURE_WEIGHT_VALUE = 1000
 
class RLAlgorithm:
    """
    :description: abstract class defining the interface of a RL algorithm
    """

    def getAction(self, state): 
        raise NotImplementedError("Override me")

    def incorporateFeedback(self, state, action, reward, newState): 
        raise NotImplementedError("Override me")

class QLearningAlgorithm(RLAlgorithm):
    """
    :description: Class implementing the Q-learning algorithm
    """
    def __init__(self, actions, discount, featureExtractor, 
                explorationProb=0.2, stepSize=0.01, maxGradient=1):
        """
        :type: actions: list
        :param actions: possible actions to take

        :type discount: float
        :param discount: the discount factor

        :type featureExtractor: callable returning dictionary 
        :param featureExtractor: returns the features extracted from a state

        :type explorationProb: float
        :param explorationProb: probability of taking a random action

        :type stepSize: float
        :param stepSize: learning rate

        :type maxGradient: float
        :param maxGradient: maximum gradient update allowed (i.e., applies gradient clipping)
        """
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 1
        self.stepSize = stepSize
        self.target_weights = self.weights
        self.maxGradient = maxGradient

    def getQ(self, state, action):
        """
        :description: returns the Q value associated with this state-action pair

        :type state: dictionary
        :param state: the state of the game

        :type action: int 
        :param action: the action for which to retrieve the Q-value
        """
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    def getAction(self, state):
        """
        :description: returns an action accoridng to epsilon-greedy policy

        :type state: dictionary
        :param state: the state of the game
        """
        self.numIters += 1
        if random.random() < self.explorationProb: 
            return random.choice(self.actions)
        else:
            maxAction = max((self.getQ(state, action), action) for action in self.actions)[1]
        return maxAction

    def getStepSize(self):
        """
        :description: return the step size
        """
        return self.stepSize

    def incorporateFeedback(self, state, action, reward, newState):
        """
        :description: performs a Q-learning update 

        :type state: dictionary
        :param state: the state of the game

        :type action: int 
        :param action: the action for which to retrieve the Q-value

        :type reward: float
        :param reward: reward associated with being in newState

        :type newState: dictionary
        :param newState: the new state of the game
        """
        stepSize = self.stepSize
        prediction = self.getQ(state, action)        
        target = reward
        if newState != None:
            target += self.discount * max((self.getQ(newState, newAction), newAction) for newAction in self.actions)[0]

        update = stepSize * (prediction - target)
        update = np.clip(update, -self.maxGradient, self.maxGradient)
        for f, v in self.featureExtractor(state, action):
            self.weights[f] = self.weights[f] - update * v
            assert(self.weights[f] < MAX_FEATURE_WEIGHT_VALUE)



