import config
from game import Game
from scipy.io import savemat, loadmat
from player import MCTSPlayer,HumanPlayer, RandomPlayer, MinMaxPlayer
from policy_value_net import PolicyValueNet
from utils import symmetry_board_moves
from rl_utils import ReplayBuffer
from tensorboardX import SummaryWriter
import numpy as np
import os
from numpy import random
from datetime import datetime
from config import TrainConfig

# Define the config for the BaghChal
training_config = TrainConfig()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

############## Stage I - Play Random Player Against Computer #########################
print('Stage I - Collecting Random and Computer Played Data')
# First Play Few Games with Goat as MinMax Player and Bagh as Random Player
datadir = os.path.join(ROOT_DIR, 'GoatAsMinMax_BaghAsRandom')
if not os.path.exists(datadir):
    os.makedirs(datadir)
NumOfGames = training_config.num_of_random_minmax_games
for gamenumber in range(NumOfGames):
    print('Playing -- Goat: MinMax, Bagh: Random, Game Number:', gamenumber + 1)
    goat = MinMaxPlayer()
    bagh = RandomPlayer()
    game = Game(depth=5)
    game.board.reset()
    start_time = int(datetime.now().timestamp())
    [w, data] = game.start_play_random_minmax(goat, bagh, show=False)
    end_time = int(datetime.now().timestamp())
    game_time = end_time - start_time
    data = [x for x in data]
    data = symmetry_board_moves(data)
    state_batch = [x[0] for x in data]
    mcts_probs_batch = [x[1] for x in data]
    winner_batch = [x[2] for x in data]
    state_batch = np.array(state_batch)
    mcts_probs_batch = np.array(mcts_probs_batch)
    winner_batch = np.array(winner_batch)
    # Keeping the game data incase we need for future
    savedict = {'Game Duration': game_time, 'Game Number': gamenumber + 1, 'state_batch': state_batch,
                'mcts_probs_batch': mcts_probs_batch,
                'winner_batch': winner_batch}
    time_string = str(int(datetime.now().timestamp()))
    filename = os.path.join(datadir, 'Game_' + str(gamenumber+1) + '_' + time_string + '.mat')
    savemat(filename, savedict)
    print('Winner:', w)
# Next Play Few Games with Goat as Random Player and Bagh as MinMax Player
datadir = os.path.join(ROOT_DIR, 'BaghAsMinMax_GoatAsRandom')
if not os.path.exists(datadir):
    os.makedirs(datadir)
for gamenumber in range(NumOfGames):
    print('Playing -- Bagh: MinMax, Goat: Random, Game Number:', gamenumber + 1)
    goat = RandomPlayer()
    bagh = MinMaxPlayer()
    game = Game(depth=5)
    game.board.reset()
    start_time = int(datetime.now().timestamp())
    [w, data] = game.start_play_random_minmax(goat, bagh, show=False)
    end_time = int(datetime.now().timestamp())
    game_time = end_time - start_time
    data = [x for x in data]
    data = symmetry_board_moves(data)
    state_batch = [x[0] for x in data]
    mcts_probs_batch = [x[1] for x in data]
    winner_batch = [x[2] for x in data]
    state_batch = np.array(state_batch)
    mcts_probs_batch = np.array(mcts_probs_batch)
    winner_batch = np.array(winner_batch)
    # Keeping the game data incase we need for future
    savedict = {'Game Duration': game_time, 'Game Numer': gamenumber + 1, 'state_batch': state_batch,
                'mcts_probs_batch': mcts_probs_batch,
                'winner_batch': winner_batch}
    time_string = str(int(datetime.now().timestamp()))
    filename = os.path.join(datadir, 'Game_' + str(gamenumber+1) + '_' + time_string + '.mat')
    savemat(filename, savedict)
    print('Winner:', w)
############ Phase II - Computer Aganist Computer ###############################
print('Phase II - Playing Computer vs Computer')
datadir = os.path.join(ROOT_DIR , 'BaghAsMinMax_GoatAsMinMax')
if not os.path.exists(datadir):
    os.makedirs(datadir)
NumOfGames = training_config.num_of_minmax_minmax_games
for depthvalue in range(4,7):
    for gamenumber in range(NumOfGames):
        print('Playing -- Bagh: MinMax, Goat: MinMax, Game Number:', gamenumber + 1, 'Tree Depth', depthvalue)
        goat = MinMaxPlayer()
        bagh = MinMaxPlayer()
        game = Game(depth=depthvalue)
        game.board.reset()
        start_time = int(datetime.now().timestamp())
        [w, data] = game.start_play_random_minmax(goat, bagh, show=False)
        end_time = int(datetime.now().timestamp())
        game_time = end_time - start_time
        data = [x for x in data]
        data = symmetry_board_moves(data)
        state_batch = [x[0] for x in data]
        mcts_probs_batch = [x[1] for x in data]
        winner_batch = [x[2] for x in data]
        state_batch = np.array(state_batch)
        mcts_probs_batch = np.array(mcts_probs_batch)
        winner_batch = np.array(winner_batch)
        # Keeping the game data incase we need for future
        savedict = {'Game Duration': game_time, 'Game Number': gamenumber + 1, 'state_batch': state_batch,
                    'mcts_probs_batch': mcts_probs_batch,
                    'winner_batch': winner_batch}
        time_string = str(int(datetime.now().timestamp()))
        filename = os.path.join(datadir, 'Game_' + str(gamenumber+1) + '_' + time_string + '.mat')
        savemat(filename, savedict)
        print('Winner:', w)