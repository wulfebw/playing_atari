import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import collections, random
import theano
import theano.tensor as T
from mlp import MLP, HiddenLayer

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
    def discount(self): return 1


# Return i in [0, ..., len(probs)-1] with probability probs[i].
def sample(probs):
    target = random.random()
    accum = 0
    for i, prob in enumerate(probs):
        accum += prob
        if accum >= target: return i
    raise Exception("Invalid probs: %s" % probs)

def test_nnet_numberline_mdp(n_episodes,  exploration_prob=0.1, learning_rate=.01):

    mdp = NumberLineMDP(5)
    actions = mdp.actions(mdp.startState())

    features = T.dvector('features')
    action = T.lscalar('action')
    reward = T.dscalar('reward')
    next_features = T.dvector('next_features')

    hidden_layer_1 = HiddenLayer(n_vis=1, n_hid=len(actions), layer_name='hidden1', activation='relu')
    # output_layer = OutputLayer(layer_name='output1', activation='relu')
    layers = [hidden_layer_1]
    mlp = MLP(layers, discount=mdp.discount(), learning_rate=learning_rate)
    loss, updates = mlp.get_loss_and_updates(features, action, reward, next_features)

    train_model = theano.function(
                    [theano.Param(features, default=np.zeros(MAX_FEATURES_TEST)),
                    theano.Param(action, default=0),
                    theano.Param(reward, default=0),
                    theano.Param(next_features, default=np.zeros(MAX_FEATURES_TEST))],
                    outputs=loss,
                    updates=updates,
                    mode='FAST_COMPILE')

    rewards = []
    for episode in xrange(n_episodes):

        cur_state = mdp.startState()
        while not mdp.isEnd(cur_state):
            print 'cur_state: {}'.format(cur_state)


            if random.random() < exploration_prob: 
                action = random.choice(actions)
            else:
                action = T.argmax(mlp.fprop([cur_state])).eval()
                print 'action: {}'.format(action)
            realAction = action
            if action == 0: realAction = -1
            
            transitions = mdp.succAndProbReward(cur_state, realAction)

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            print transitions[i]
            newState, prob, reward = transitions[i]
            print 'newState: {}'.format(newState)
            print 'reward: {}'.format(reward)
            print [(p.eval(), p.name) for p in mlp.get_params()]



            loss = train_model([cur_state], action, reward, [newState])
            print 'loss: {}'.format(loss)
            print '\n'

            cur_state = newState
            
            # if verbose and counter % 53 == 0:
            #     print('*' * 15 + ' training information ' + '*' * 15) 
            #     print('episode: {}'.format(episode))
            #     print('reward: \t{}'.format(reward))
            #     print('action: \t{}'.format(real_actions[action]))
            #     param_info = [(p.eval(), p.name) for p in mlp.get_params()]
            #     for index, (val, name) in enumerate(param_info):
            #         if previous_param_0 is None and index == 0:
            #             previous_param_0 = val
            #         print('parameter {} value: \n{}'.format(name, val))
            #         if index == 0:
            #             diff = val - previous_param_0
            #             print('difference from previous param {}: \n{}'.format(name, diff))
            #     print('features: \t{}'.format(features))
            #     print('next_features: \t{}'.format(next_features))
            #     print('*' * 52)
            #     print('\n')

            rewards.append(reward)

        
        print('episode: {} ended with score: {}'.format(episode, rewards[-1]))
        
    return rewards

if __name__ == '__main__':
    test_nnet_numberline_mdp(10000)
    


