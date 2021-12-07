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
import pandas as pd
from config import TrainConfig

# Retrive Root Directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define Game and Players
game = Game(depth=4)
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
            integer_map = map(int, temp_game_number)
            integer_list = list(integer_map)
            sorted_list = sorted(integer_list)
            return str(sorted_list[-1])
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
            integer_map = map(int, temp_date_time)
            integer_list = list(integer_map)
            sorted_list = sorted(integer_list, reverse=True)
            latest_model_file = 'model_selfplay_' + str(sorted_list[0]) + '.h5'
            return latest_model_file
        else:
            print(f"No pretrain model exits for current configuration, pre_epochs {game_config.pre_epochs}, epoch "
                  f"{game_config.epochs}, simulations {game_config.n_playout}")
    except Exception as exp:
        print("Exception occured ", exp)
        raise exp

def evaluate_game(pvnet_value_fn, model_path, player='Minmax'):
    print(f"Playing with {player} Player")
    if player == 'MCTS':
        pvnet_untrained = PolicyValueNet()
        pvnet_untrained.loss_train_op()
        pvnet_untrained_value_fn = pvnet_untrained.policy_value_fn
    while True:
        player_list = ["G", "B"]
        choice = random.choice(player_list)
        if choice == 'B':
            if player=='MCTS':
                goat = MCTSPlayer(pvnet_untrained_value_fn, n_playout=game_config.n_playout, is_selfplay=0)
            elif player=='Minmax':
                goat = MinMaxPlayer()
            else:
                goat = RandomPlayer()

            bagh = MCTSPlayer(pvnet_value_fn, n_playout=game_config.n_playout, is_selfplay=0)
            break
        elif choice == 'G':
            if player=='MCTS':
                bagh = MCTSPlayer(pvnet_untrained_value_fn, n_playout=game_config.n_playout, is_selfplay=0)
            elif player=='Minmax':
                bagh = MinMaxPlayer()
            else:
                bagh = RandomPlayer()

            goat = MCTSPlayer(pvnet_value_fn, n_playout=game_config.n_playout, is_selfplay=0)
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
    return choice, w, game_time

def evaluate_baghchal_game(player='Minmax'):
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

    datadir = os.path.join(ROOT_DIR, 'Evaluate_Data')

    data_mode = 'self_play_only'

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    temp_data = []

    for game in range(100):
        player1, winner_player, game_time = evaluate_game(pvnet_value_fn, model_path, player='Minmax')
        temp_data.append(['Minmax', player1, winner_player, game_time])

    input_minmax_dataframe = pd.DataFrame(temp_data, columns=('Algorithm', 'Player_Chosen', 'Winner_Player', 'Game_Time'))
    pd.DataFrame.to_csv(input_minmax_dataframe, os.path.join(datadir, f'play_mcts_with_minmax_{data_mode}.csv'))


    temp_data = []
    for game in range(100):
        player1, winner_player, game_time = evaluate_game(pvnet_value_fn, model_path, player='MCTS')
        temp_data.append(['MCTS', player1, winner_player, game_time])

    input_dataframe = pd.DataFrame(temp_data, columns=('Algorithm', 'Player_Chosen', 'Winner_Player', 'Game_Time'))
    pd.DataFrame.to_csv(input_dataframe, os.path.join(datadir, f'play_mcts_with_untrained_mcts_{data_mode}.csv'))


    temp_data = []
    for game in range(100):
        player1, winner_player, game_time = evaluate_game(pvnet_value_fn, model_path, player='Random')
        temp_data.append(['Random', player1, winner_player, game_time])

    input_randomplayer_dataframe = pd.DataFrame(temp_data, columns=('Algorithm', 'Player_Chosen', 'Winner_Player', 'Game_Time'))
    pd.DataFrame.to_csv(input_randomplayer_dataframe, os.path.join(datadir, f'play_mcts_randomplayer_{data_mode}.csv'))

    # evaluate_game(pvnet_value_fn, model_path, player='Random')
    # evaluate_game(player='MCTS')
    # evaluate_game(player='RandomPlayer')

evaluate_baghchal_game()