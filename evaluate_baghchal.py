import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
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

# Retrive Root Directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define Game and Players
game = Game()
game_config = TrainConfig()
# Load trained Policy-Value Network
self_play_model_directory = os.path.join(ROOT_DIR, 'models', 'model', 'self_play_model',
                                                     f'model_selfplay_{game_config.epochs}_epoch_{game_config.n_playout}_simulations_{99 + 1}_gamenumber')
model_path = os.path.join(self_play_model_directory, 'model_selfplay_1638576146.h5')
if not os.path.exists(model_path):
    print('No trained Model is Found')
    exit(0)
else:
    pvnet = PolicyValueNet(model_path)

# Define Players
pvnet.loss_train_op()
pvnet_value_fn = pvnet.policy_value_fn
while True:
    choice = input('Play as Goat (G) or Tiger (T) (Typer G or T):')
    choice = choice.upper()
    if choice == 'T':
        goat = MinMaxPlayer()
        bagh = MCTSPlayer(pvnet_value_fn, n_playout=game_config.n_playout, is_selfplay=0)
        break
    elif choice == 'G':
        goat = MCTSPlayer(pvnet_value_fn, n_playout=game_config.n_playout, is_selfplay=0)
        bagh = MinMaxPlayer()
        break
    else:
        print('Invalid Player! Select either G or T')
# Start the playing game
game.board.reset()
start_time = int(datetime.now().timestamp())
[w, data] = game.start_play(goat, bagh, show=True)
end_time = int(datetime.now().timestamp())
game_time = end_time - start_time
print('Game Over!', 'Winner: ', w, 'Game Time:', game_time)
