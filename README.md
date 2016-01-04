Project Overview
----------------

The original goal of this project was to develop an agent capable of playing many different Atari games using the screen pixel data and score. We were largely inspired by the deepmind papers attempting the same task [0][1], and hoped to answer a few questions we had about that research. 

Questions + Conclusions
-----------------------

1. <b>Question:</b> Is it possible to use traditional object detection/classification/tracking methods to derive features from the screen which can then be passed into a Deep Q-Network (DQN) and achieve good performance? These features would clearly be less informative than those learned by a Convolutional Nerual Network (CNN), so we expected to achieve relatively worse performance, though at significantly reduced computation time.

  <b>Conclusion:</b> Using traditional feature extraction methods with a DQN beats the baseline but is much worse than the CNN based DQN. There are a number of reasons this could be the case. One might be that the object features we extracted do not contain sufficient information to achieve high performance on Atari games.

2. <b>Question:</b> How important is experience replay to the DQN success and how do the hyperparameters of the replay memory impact performance? 

   <b>Conclusion:</b> The experience replay makes a tremendous difference and so do the replay hyperparameters (we evaluated on smaller MDPs, but believe these results apply in the case of Atari games as well). Figure 1 shows the performance of the different replay memories keeping the number of trials constant and figure 2 shows performance of different replay memories keeping the number of updates constant (here updates are really what matters). From these results we concluded that larger capacity replay memories perform much better (likely because the samples taken from these memories are not as highly correlated with each other) and that larger updates per trial also perform better.

Figure 1: Constant number of trials, green is the largest capacity/update replay memory and blue is no memory
![alt text](https://raw.githubusercontent.com/wulfebw/playing_atari/master/media/const_trials.png)

Figure 2: Constant number of updates, green is the largest capacity/update replay memory and blue is no memory
![alt text](https://raw.githubusercontent.com/wulfebw/playing_atari/master/media/const_updates.png)
  
3. <b>Question:</b> How important is it to use a stationary target function for the Q-learning updates?

  <b>Conclusion:</b> It is really important. If you do not use a semi-static target function then weights will tend to diverge with both linear and neural network function approximators (at the same and much lower learning rates).
  
Results
-------
We developed Sarsa(lambda) and Q-learning models with both linear and neural network function approximators. Using the object-based features, Sarsa(lambda) with linear function approximation achieved the highest performance both on Breakout and on the games in the test group. It is interesting to visualize what the model learned by plotting the maximum Q-value achievable depending upon the location of the ball in breakout (figure 3).

Figure 3: Optimal Q value in the game Breakout, dark red indicates high Q value and dark blue represents low Q value
![alt tag](https://raw.githubusercontent.com/wulfebw/playing_atari/master/media/heatmap.png)

What's next?
------------

Clearly it's possible to play breakout well (we used an existing theano implementation of the DQN to train this model [2]):
![alt tag](https://raw.githubusercontent.com/wulfebw/playing_atari/master/media/breakout.gif "Playing Breakout")  

More difficult is learning to play somewhat complex games like Montezuma's Revenge:
![alt tag](https://raw.githubusercontent.com/wulfebw/playing_atari/master/media/montezuma_revenge_no_intrinsic.gif "Montezuma's Revenge Without Intrinsic Rewards")  

<b> The next goal is to develop an agent that can play Montezuma's Revenge and other games of similar difficulty. </b>

How to do that?
---------------

Among the challenges to playing these types of games is that the rewards are sparse. For example, it is highly unlikely that the explorer in Montezuma's Revenge will manage to reach the key through random actions (he would have to climb up/down 3 ladders and jump over the skeleton before receiving any sort of reward signal). We're going to try to overcome this challenge using intrinsic rewards [3]. Specifically, we're going to implement some form of artificial curiosity [4] as well as some form of empowerment [5]. Curiosity basically motivates an agent to explore the state space. Empowerment basically motivates an agent to maximize its influence over the environment.

References
----------
[0] Deepmine Nips paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

[1] Deepmind Nature Paper: http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html

[2] DQN thenao implementation: https://github.com/spragunr/deep_q_rl

[3] Intrinsic motivation Paper: http://papers.nips.cc/paper/2552-intrinsically-motivated-reinforcement-learning.pdf

[4] Curiosity: http://people.idsia.ch/~juergen/interest.html

[5] Empowerment: http://arxiv.org/pdf/1310.1863v2.pdf

Other Resources
---------------

[6] Our report: http://cs229.stanford.edu/proj2015/366_report.pdf 

[7] Arcade learning environment (ALE) paper: http://arxiv.org/pdf/1207.4708v2.pdf

[8] ALE website: http://www.arcadelearningenvironment.org/

[9] Basic background paper on playing Atari games: http://arxiv.org/pdf/1410.8620v1.pdf

[10] Recent empowerment research: http://arxiv.org/pdf/1509.08731v1.pdf


