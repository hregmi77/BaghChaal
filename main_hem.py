from game import Game
from scipy.io import savemat, loadmat
from player import MCTSPlayer,HumanPlayer
from policy_value_net import PolicyValueNet
from utils import symmetry_board_moves
import numpy as np
game=Game()
goat = HumanPlayer()
bagh = HumanPlayer()
for GameNumber in range(10):
    data=game.start_play(bagh,goat)
    data=[x for x in data]
    data=symmetry_board_moves(data)
    state_batch = [x[0] for x in data]
    mcts_probs_batch = [x[1] for x in data]
    winner_batch = [x[2] for x in data]
    # Save data for training
    # GameNumber = 1
    savedict = {'Game Numer': GameNumber+1, 'state_batch': state_batch, 'mcts_probs_batch': mcts_probs_batch,
                'winner_batch': winner_batch}
    filename = 'C:/Users/hregmi/PycharmProjects/BaghChaalProject/humanplayerdata/Game_' + str(GameNumber) + '.mat'
    savemat(filename, savedict)
# # Load data for training
# datafile = loadmat(filename)
# state_batch = datafile['state_batch']
# mcts_probs_batch = datafile['mcts_probs_batch']
# winner_batch = datafile['winner_batch']
# winner_batch = np.squeeze(winner_batch)
# pvnet.train(state_batch,mcts_probs_batch,winner_batch,50)
# pvnet.save_model("model.h5")

