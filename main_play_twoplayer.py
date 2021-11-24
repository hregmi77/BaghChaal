from game import Game
from scipy.io import savemat, loadmat
from player import MCTSPlayer,HumanPlayer
from policy_value_net import PolicyValueNet
from utils import symmetry_board_moves
import numpy as np
import os
game = Game()
goat = HumanPlayer()
bagh=HumanPlayer()

root_dir = os.path.dirname(os.path.dirname(os.path.abspath('__filename__')))


# Start Data Collection and Network training Process
continue_game = 'yes'

data_dir = os.path.join(root_dir, 'humanplayer_data_minmax')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

game_counter = len(os.listdir(data_dir)) + 1
while continue_game == 'yes':
    game.board.reset()
    data = game.start_play(goat, bagh)
    data=[x for x in data]
    data=symmetry_board_moves(data)
    state_batch = [x[0] for x in data]
    mcts_probs_batch = [x[1] for x in data]
    winner_batch = [x[2] for x in data]
    # Save data for training
    savedict = {'Game Number': game_counter, 'state_batch': state_batch, 'mcts_probs_batch': mcts_probs_batch,
                'winner_batch': winner_batch}

    filename = data_dir + 'Game_' + str(game_counter) + '.mat'
    savemat(filename, savedict)
    continue_game = input('contine game (yes/no):')
    game_counter += 1
