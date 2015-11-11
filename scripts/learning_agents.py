import sys, collections, math, random
import pprint

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")


# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 1

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
	#print "############# checking actions: " + str(len(self.actions))
        if random.random() < self.explorationProb: 
            return random.choice(self.actions)
        else:
            maxAction = max((self.getQ(state, action), action) for action in self.actions)[1]
	return maxAction

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        #return 1.0 / self.numIters
        return .001

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # here's how q-learning works
        # it's an off-model bootstrapping method
        
        # offmodel: this means it doesn't even try to predict what the transition or reward 
        # probabilities are, all it does is try to directly predict the value function
        # specifically, it tries to predict the optimal value function (whereas 
        # SARSA tries to predict the value function of a given policy)
        
        # bootstrapping: means it uses the current appromixation to evaluate future approximations
        
        # 1. calculate the current _prediction_ for the Qopt value using getQ
        # 2. calculate a _target_ value also using Qopt, but do so over the next possible states and actions
        #    also adding the reward we just got
        # 3. if these two values are far apart, update the weights a lot, o/w don't update them too much


        stepSize = self.getStepSize()
        prediction = self.getQ(state, action)        
        target = reward
        if newState != None:
            target += self.discount * max((self.getQ(newState, newAction), newAction) for newAction in self.actions)[0]

        for f, v in self.featureExtractor(state, action):
            # print("feature: {}\textracted value: {}".format(f,v))
            self.weights[f] = self.weights[f] - stepSize*(prediction - target)*v
            assert(self.weights[f] < 1000000)


        # print('prediction: {}'.format(prediction))
        # print('target: {}'.format(target))
        # print('state["objects"]: {}'.format(state["objects"]))
        # print('action: {}'.format(action))
        # print('reward: {}'.format(reward))
        # print('newState["objects"]: {}'.format(newState["objects"]))
        # for k, v in self.weights.iteritems():
        #     print('feature: {}\t weight: {}'.format(k,v))
        # print '\n' * 5

