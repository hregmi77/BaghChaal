import os.path
from datetime import datetime
from game import Game
from scipy.io import savemat, loadmat
from player import MCTSPlayer,HumanPlayer
from policy_value_net import PolicyValueNet
from utils import symmetry_board_moves
import numpy as np

# Retrive Root Directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(ROOT_DIR, 'humanplayerdata')
if not os.path.exists(datadir):
    os.makedirs(datadir)
game=Game()
goat = HumanPlayer()
bagh = HumanPlayer()
continue_game = 'YES'
while continue_game == 'YES':
    start_time = int(datetime.now().timestamp())
    [w, data] = game.start_play(goat, bagh, show=True)
    end_time = int(datetime.now().timestamp())
    game_time = end_time - start_time
    data=[x for x in data]
    data=symmetry_board_moves(data)
    state_batch = [x[0] for x in data]
    mcts_probs_batch = [x[1] for x in data]
    winner_batch = [x[2] for x in data]
    # Save data for training
    # Keeping the game data incase we need for future
    savedict = {'Game Duration': game_time, 'Game Numer': gamenumber + 1, 'state_batch': state_batch,
                'mcts_probs_batch': mcts_probs_batch,
                'winner_batch': winner_batch}
    time_string = str(int(datetime.now().timestamp()))
    filename = os.path.join(datadir, 'Game_' + str(gamenumber+1) + '_' + time_string + '.mat')
    savemat(filename, savedict)
    continue_game = input('Contine Game (Yes/No)')
    continue_game.upper()
