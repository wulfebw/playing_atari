import os, sys, copy 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import collections, random
import theano
import theano.tensor as T
from qnetwork import QNetwork, HiddenLayer, OutputLayer

import numpy as np

MAX_FEATURES_TEST = 1

# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        # print "%d states" % len(self.states)
        # print self.states

############################################################

# A simple example of an MDP where states are integers in [-n, +n].
# and actions involve moving left and right by one position.
# We get rewarded for going to the right.
class NumberLineMDP(MDP):
    def __init__(self, n=5): self.n = n
    def startState(self): return 0
    def isEnd(self, state): return state == self.n 
    def actions(self, state): return [-1, +1]
    def succAndProbReward(self, state, action):
        return [(state, 0.4, 0),
                (min(max(state + action, -self.n), +self.n), 0.6, state)]
    def discount(self): return .9

###########################################################

class GridSearchMDP(MDP):
    def __init__(self, n=10):
	self.n = n
    def startState(self): return (random.randint(2, self.n - 4) + 0.0, random.randint(2, self.n - 4) + 0.0)
    def isEnd(self, state): return state[0] == 0 or state[0] >= self.n or state[1] == 0 or state[1] >= self.n
    def actions(self, state): return [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1)]
    def succAndProbReward(self, state, action):
	newState = (state[0] + action[0], state[1] + action[1])
	nextReward = 0
	if self.isEnd(newState):
		if newState[0] == 0 or newState[1] == 0 or (newState[0] == self.n and newState[1] != self.n):
			nextReward = -1
		elif newState[0] == self.n and newState[1] == self.n:
			nextReward = 5
		elif newState[1] == self.n:
			nextReward = 1
	return [(state, 0.4, 0), (newState, 0.6, nextReward)]
    def discount(self): return 0.99

# Return i in [0, ..., len(probs)-1] with probability probs[i].
def sample(probs):
    target = random.random()
    accum = 0
    for i, prob in enumerate(probs):
        accum += prob
        if accum >= target: return i
    raise Exception("Invalid probs: %s" % probs)

def test_nnet_numberline_mdp(n_episodes,  exploration_prob=0.9, learning_rate=.0005, target_freeze_period=500):

    reduce_explore = 0.0001
    size = 20.
    mdp = GridSearchMDP(size)
    actions = mdp.actions(mdp.startState())

    print actions

    features = T.dvector('features')
    action = T.lscalar('action')
    reward = T.dscalar('reward')
    next_features = T.dvector('next_features')

    n_vis = 2 # for chain mdp
    hidden_layer_1 = HiddenLayer(n_vis=n_vis, n_hid=len(actions), layer_name='hidden', activation='tanh')
    output_layer = OutputLayer(layer_name='out', activation='relu')
    layers = [hidden_layer_1, output_layer]
    mlp = QNetwork(layers, discount=mdp.discount(), learning_rate=learning_rate)
    loss, updates = mlp.get_loss_and_updates(features, action, reward, next_features)

    train_model = theano.function(
                    [theano.Param(features, default=np.zeros(MAX_FEATURES_TEST)),
                    theano.Param(action, default=0),
                    theano.Param(reward, default=0),
                    theano.Param(next_features, default=np.zeros(MAX_FEATURES_TEST))],
                    outputs=loss,
                    updates=updates,
                    mode='FAST_RUN')

    rewards = []
    counter = 0
    for episode in xrange(n_episodes):
	curDiscount = mdp.discount()
	totalReward = 0
        cur_state = mdp.startState()
	print cur_state
        while not mdp.isEnd(cur_state):
            counter += 1
            if counter % 1000 == 0:
                mlp.frozen_layers = copy.deepcopy(mlp.layers)

	    if (counter % 100 == 0):
            	print 'cur_state: {}'.format(cur_state)

            if random.random() < exploration_prob: 
                action = random.choice(actions)
                action_index = actions.index(action)
            else:
                action_index = T.argmax(mlp.fprop([cur_state[0]/mdp.n, cur_state[1]/mdp.n])).eval()
                action = actions[action_index]
		if (counter % 100 == 0):
               	    print 'action: {}'.format(action)
            # realAction = action
            # if action == 0: realAction = -1
            transitions = mdp.succAndProbReward(cur_state, action) # previously realAction)
            if len(transitions) == 0:
                break
            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            #print 'newState: {}'.format(newState)
            #print 'reward: {}'.format(reward)
            #print [(p.eval(), p.name) for p in mlp.get_params()]
            #print [(p.eval(), p.name) for p in mlp.get_params(freeze=True)]
            #print '\n'
	    reward *= curDiscount
	    totalReward += reward
	    curDiscount *= mdp.discount()
            loss = train_model([cur_state[0]/mdp.n, cur_state[1]/mdp.n], action_index, reward, [newState[0]/mdp.n, newState[1]/mdp.n]) # previously action
            cur_state = newState
	    exploration_prob -= reduce_explore
	    if (exploration_prob < 0.25):
		exploration_prob = 0.25
        rewards.append(totalReward)
	
        print('*' * 30)
        print('episode: {} ended with score: {}'.format(episode, rewards[-1]))
	print('avg reward: {}'.format(np.mean(rewards[-25:])))
	print('explore: {}'.format(exploration_prob))
        print('*' * 30)
        print('\n')
        
    return rewards

if __name__ == '__main__':
    test_nnet_numberline_mdp(10000)
    


