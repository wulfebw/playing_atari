"""
:description: get baseline random performance on set of atari 2600 games
"""

import os
import sys
from random import randrange
from ale_python_interface import ALEInterface



def get_random_baseline(gamepath):
    ale = ALEInterface()
    ale.setInt('random_seed', 42)

    recordings_dir = './recordings/breakout/'

    USE_SDL = True
    if USE_SDL:
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            ale.setBool('sound', False) # Sound doesn't work on OSX
            #ale.setString("record_screen_dir", recordings_dir);
        elif sys.platform.startswith('linux'):
            ale.setBool('sound', True)
            ale.setBool('display_screen', True)

    # Load the ROM file
    ale.loadROM(gamepath)

    # Get the list of legal actions
    legal_actions = ale.getLegalActionSet()

    # Play 5 episodes
    rewards = []
    for episode in xrange(10):
        total_reward = 0
        while not ale.game_over():
            a = legal_actions[randrange(len(legal_actions))]
            reward = ale.act(a);
            total_reward += reward
        rewards.append(total_reward)
        #print 'Episode', episode, 'ended with score:', total_reward
        ale.reset_game()
    avg_reward = sum(rewards) / float(len(rewards))
    return avg_reward

if __name__ == '__main__':

    # not working: 'beamrider.bin', 'q-bert.bin', 'stargunner.bin', 'james_bond.bin', 'ms_pac_man.bin'
    base_dir = '/Users/wulfe/Dropbox/School/Stanford/autumn_2015/cs221/project/Roms/ROMS'
    games = ['freeway.bin', 'breakout.bin', 'space_invaders.bin', 'robotank.bin', 
            'road_runner.bin', 'ice_hockey.bin', 'asterix.bin', 'chopper_command.bin',
            'crazy_climber.bin', 'kangaroo.bin', 'bank_heist.bin', 'defender.bin',
            'krull.bin']
    baselines = []
    for game in games:
        gamepath = os.path.join(base_dir, game)
        baseline = get_random_baseline(gamepath)
        baselines.append(baseline)
    print('\n')
    for game, baseline in zip(games, baselines):
        print('game: {}\t\tbaseline: {}'.format(game, baseline))
