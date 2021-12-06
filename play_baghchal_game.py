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


def get_latest_game_number(self_play_model_directory):
    try:
        if os.path.exists(self_play_model_directory):
            dir_list = os.listdir(self_play_model_directory)
            temp_game_number = []
            for dirs in dir_list:
                if f'model_selfplay_{game_config.epochs}_epoch_{game_config.n_playout}_simulations' in dirs:
                    temp_game_number.append(dirs.split('_')[-2])
            return sorted(temp_game_number)[-1]
    except Exception as exp:
        print(f"Exception {exp} occured")

def get_latest_model_file(self_play_model_directory):
    try:
        if os.path.exists(self_play_model_directory):
            files = os.listdir(self_play_model_directory)
            temp_date_time = []
            for file in files:
                if file.endswith(".h5"):
                    temp_date_time.append(file.split('.')[0].split('_')[-1])
            temp_date_time = sorted(temp_date_time, reverse=True)
            latest_model_file = 'model_selfplay_' + temp_date_time[0] + '.h5'
            return latest_model_file
        else:
            print(f"No pretrain model exits for current configuration, pre_epochs {game_config.pre_epochs}, epoch "
                  f"{game_config.epochs}, simulations {game_config.n_playout}")
    except Exception as exp:
        print("Exception occured ", exp)
        raise exp

def play_game(pvnet_value_fn, model_path, player='minmax'):
    print(f"Playing with {player} Player")
    if player == 'mcts':
        pvnet_untrained = PolicyValueNet(model_path)
        pvnet_untrained.loss_train_op()
        pvnet_untrained_value_fn = pvnet_untrained.policy_value_fn
    while True:
        choice = input('Play as Goat (G) or Tiger (B) (Typer G or B):')
        choice = choice.upper()
        if choice == 'B':
            if player=='mcts':
                goat = MCTSPlayer(pvnet_untrained_value_fn, n_playout=game_config.n_playout, is_selfplay=0)
            elif player=='minmax':
                goat = MinMaxPlayer()
            else:
                goat = RandomPlayer()

            bagh = MCTSPlayer(pvnet_value_fn, n_playout=game_config.n_playout, is_human_play=1, is_selfplay=0)
            break
        elif choice == 'G':
            if player=='mcts':
                bagh = MCTSPlayer(pvnet_untrained_value_fn, n_playout=game_config.n_playout, is_selfplay=0)
            elif player=='minmax':
                bagh = MinMaxPlayer()
            else:
                bagh = RandomPlayer()

            goat = MCTSPlayer(pvnet_value_fn, n_playout=game_config.n_playout, is_human_play=1, is_selfplay=0)
            break
        else:
            print('Invalid Player! Select either G or B')
    # Start the playing game
    game.board.reset()
    start_time = int(datetime.now().timestamp())
    [w, data] = game.start_play(goat, bagh, show=True)
    end_time = int(datetime.now().timestamp())
    game_time = end_time - start_time
    print('Game Over!', 'Winner: ', w, 'Game Time:', game_time)

def play_baghchal():
    self_play_model_directory = os.path.join(ROOT_DIR, 'models', 'model', 'self_play_model')
    gamenumber = get_latest_game_number(self_play_model_directory)
    self_play_model_directory = os.path.join(ROOT_DIR, 'models', 'model', 'self_play_model',
                                                         f'model_selfplay_{game_config.epochs}_epoch_{game_config.n_playout}_simulations_{gamenumber}_gamenumber')
    latest_model_file = get_latest_model_file(self_play_model_directory)
    model_path = os.path.join(self_play_model_directory, latest_model_file)
    if not os.path.exists(model_path):
        print('No trained Model is Found')
        exit(0)
    else:
        pvnet = PolicyValueNet(model_path)

    # Define Players
    pvnet.loss_train_op()
    pvnet_value_fn = pvnet.policy_value_fn

    player_list = ['minmax', 'random', 'mcts']
    select_player = True

    while select_player:
        player = input('Enter the Player with whom you want to play,'
                       'Minmax for Minmax player, Random for Random Player, MCTS for MCTS player:')
        player = player.lower()

        if player in player_list:
            select_player = False
            play_game(pvnet_value_fn, model_path, player=player)
        else:
            print('Invalid Player! Select either Minmax , MCTS or Random')