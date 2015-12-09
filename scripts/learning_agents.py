"""
NOTE: Parts of this file are adapted from the Stanford class CS221 Artificial Intelligence.
"""

import sys, collections, math, random
import numpy as np

from eligibility_traces import EligibilityTraces

MAX_FEATURE_WEIGHT_VALUE = 1000
 
class RLAlgorithm(object):
    """
    :description: abstract class defining the interface of a RL algorithm
    """

    def getAction(self, state): 
        raise NotImplementedError("Override me")

    def incorporateFeedback(self, state, action, reward, newState): 
        raise NotImplementedError("Override me")

class ValueLearningAlgorithm(RLAlgorithm):
    """
    :description: base class for RL algorithms that approximate the value function.
    """
    def __init__(self, actions, discount, featureExtractor, 
                explorationProb, stepSize, maxGradient=1,
                num_consecutive_random_actions=0):
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

        :type num_consecutive_random_actions: int
        :param num_consecutive_random_actions: number of times to repeat a random action
        """
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 1
        self.stepSize = stepSize
        self.maxGradient = maxGradient

        self.num_consecutive_random_actions = num_consecutive_random_actions
        self.cur_random_action_streak = 0
        self.cur_random_action = self.actions[0]

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

        if self.cur_random_action_streak > 0:
            self.cur_random_action_streak -= 1
            return self.cur_random_action

        if random.random() < self.explorationProb: 
            self.current_random_action_streak = self.num_consecutive_random_actions
            self.cur_random_action = random.choice(self.actions)
            return self.cur_random_action
        else:
            maxAction = max((self.getQ(state, action), action) for action in self.actions)[1]
        return maxAction

    def getStepSize(self):
        """
        :description: return the step size
        """
        return self.stepSize

    def incorporateFeedback(self, state, action, reward, newState): 
        raise NotImplementedError("Override me")

class QLearningAlgorithm(ValueLearningAlgorithm):
    """
    :description: Class implementing the Q-learning algorithm
    """
    def __init__(self, actions, discount, featureExtractor, 
                explorationProb, stepSize, maxGradient=1):
        """
        :note: please see parent class for params not described here
        """
        super(QLearningAlgorithm, self).__init__(actions, discount, featureExtractor, 
                    explorationProb, stepSize, maxGradient)

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

        :type rval: int or None
        :param rval: if rval returned, then this is the next action taken
        """
        stepSize = self.stepSize
        prediction = self.getQ(state, action)        
        target = reward
        if newState != None:
            target += self.discount * max(self.getQ(newState, newAction) for newAction in self.actions)

        update = stepSize * (prediction - target)
        update = np.clip(update, -self.maxGradient, self.maxGradient)
        for f, v in self.featureExtractor(state, action):
            self.weights[f] = self.weights[f] - update * v
            assert(self.weights[f] < MAX_FEATURE_WEIGHT_VALUE)
        # return None to denote that this is a off-policy algorithm
        return None

class SARSALearningAlgorithm(ValueLearningAlgorithm):
    """
    :description: Class implementing the SARSA algorithm
    """
    def __init__(self, actions, discount, featureExtractor, 
                explorationProb, stepSize, maxGradient=1):
        """
        :note: please see parent class for params not described here
        """
        super(SARSALearningAlgorithm, self).__init__(actions, discount, featureExtractor, 
                    explorationProb, stepSize, maxGradient)

    def incorporateFeedback(self, state, action, reward, newState):
        """
        :description: performs a SARSA update 

        :type state: dictionary
        :param state: the state of the game

        :type action: int 
        :param action: the action for which to retrieve the Q-value

        :type reward: float
        :param reward: reward associated with being in newState

        :type newState: dictionary
        :param newState: the new state of the game

        :type rval: int or None
        :param rval: if rval returned, then this is the next action taken
        """
        stepSize = self.stepSize
        prediction = self.getQ(state, action)        
        target = reward
        newAction = None
        if newState != None:
            # SARSA differs from Q-learning in that it does not take the max
            # over actions, but instead selects the action using it's policy
            # and in that it returns the action selected
            # so that the main training loop may use that in the next iteration
            newAction = self.getAction(newState)
            target += self.discount * self.getQ(newState, newAction)

        update = stepSize * (prediction - target)
        update = np.clip(update, -self.maxGradient, self.maxGradient)
        for f, v in self.featureExtractor(state, action):
            self.weights[f] = self.weights[f] - update * v
            assert(self.weights[f] < MAX_FEATURE_WEIGHT_VALUE)
        return newAction


class SARSALambdaLearningAlgorithm(ValueLearningAlgorithm):
    """
    :description: Class implementing the SARSA Lambda algorithm. This 
        class is equivalent to the SARSALearningAlgorithm class when
        self.lambda is set to 0; however, we keep it separate here
        because it imposes an overhead of tracking eligibility
        traces and because it is nice to see the difference between
        the two clearly.
    """
    def __init__(self, actions, discount, featureExtractor, 
                explorationProb, stepSize, threshold, decay, maxGradient=1):
        """
        :note: please see parent class for params not described here
        """
        super(SARSALambdaLearningAlgorithm, self).__init__(actions, discount, featureExtractor, 
                    explorationProb, stepSize, maxGradient)
        self.eligibility_traces = EligibilityTraces(threshold, decay)

    def incorporateFeedback(self, state, action, reward, newState):
        """
        :description: performs a SARSA update 

        :type state: dictionary
        :param state: the state of the game

        :type action: int 
        :param action: the action for which to retrieve the Q-value

        :type reward: float
        :param reward: reward associated with being in newState

        :type newState: dictionary
        :param newState: the new state of the game

        :type rval: int or None
        :param rval: if rval returned, then this is the next action taken
        """
        stepSize = self.stepSize
        prediction = self.getQ(state, action)
        self.eligibility_traces.update_all()        
        target = reward
        newAction = None
        if newState != None:
            # SARSA differs from Q-learning in that it does not take the max
            # over actions, but instead selects the action using it's policy
            # and in that it returns the action selected
            # so that the main training loop may use that in the next iteration
            newAction = self.getAction(newState)
            target += self.discount * self.getQ(newState, newAction)

        update = stepSize * (prediction - target)
        update = np.clip(update, -self.maxGradient, self.maxGradient)

        for f, v in self.featureExtractor(state, action):
            ### v might actually be 1 ###
            self.eligibility_traces[f] += v

        for f, e in self.eligibility_traces.iteritems():
            #print 'update * e: {} applied to {}, e: {}, update: {}'.format(-1 * update * e, f, e, update)
            self.weights[f] -= update * e
            assert(self.weights[f] < MAX_FEATURE_WEIGHT_VALUE)

        return newAction

