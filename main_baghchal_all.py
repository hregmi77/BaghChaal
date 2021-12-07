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
import pandas as pd

# Define the config for the BaghChal
training_config = TrainConfig()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def ComputeEloRating(Player1_Rating, Player2_Rating, Winner, K):
    '''
https://metinmediamath.wordpress.com/2013/11/27/how-to-calculate-the-elo-rating-including-example/
    :param Player1_Rating: Bagh Player Current Rating
    :param Player2_Rating: Goat Player Current Rating
    :param Winner: Whoever wins the game
    :param K: factor to effect the game result into rating, in chess, K = 32
    :return: updated ratings of both player1 and player2
    '''
    r1:float = np.power(10, Player1_Rating/400)
    r2:float = np.power(10, Player2_Rating/400)
    # compute expected score
    e1 = r1 / (r1 + r2)
    e2 = r2 / (r1 + r2)
    # compute actual score based on win or lose
    if Winner == 'G':
        s1 = 0
        s2 = 1
    elif Winner == 'B':
        s2 = 0
        s1 = 1
    else:
        s1 = 0.5
        s2 = 0.5
    # Compute the updated rating
    Player1_Rating_Updated:float = Player1_Rating + (K * (s1 - e1))
    Player2_Rating_Updated:float = Player2_Rating + (K * (s2 - e2))

    return Player1_Rating_Updated, Player2_Rating_Updated

################# To log the data in tensorboard #########################
# time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# log_dir = os.path.join('log_mcts_selfplay_50000', 'train', time_string)

# if not os.path.exists(log_dir): os.makedirs(log_dir)
# writer = SummaryWriter(log_dir=os.path.join( log_dir, 'tensorboard'))
############################################################################
# Define Buffer to store the game experience
# replaybuffer = ReplayBuffer(training_config.buffer_size)
############## Phase II - Load Previously Collected Data to Buffer #########################

def load_collected_data_to_buffer(replaybuffer):
    print('------ Load Collected Data to the Buffer  -----')
    GameFolders = ["GoatAsMinMax_BaghAsRandom", "BaghAsMinMax_GoatAsRandom", "BaghAsMinMax_GoatAsMinMax",
                   "humanplayerdata"]
    states_batchs = []
    mcts_probs_batches = []
    winner_batches = []
    counter = 1
    for gamefolder in GameFolders:
        eachfolder = os.path.join(ROOT_DIR, gamefolder)
        if os.path.isdir(eachfolder):
            for gamefile in os.listdir(eachfolder):
                filename = os.path.join(eachfolder, gamefile)
                datafile = loadmat(filename)
                state_batch = datafile['state_batch']
                mcts_probs_batch = datafile['mcts_probs_batch']
                winner_batch = datafile['winner_batch']
                winner_batch = np.squeeze(winner_batch)
                if counter == 1:
                    counter = 2
                    states_batchs = state_batch
                    mcts_probs_batches = mcts_probs_batch
                    winner_batches = winner_batch
                states_batchs = np.concatenate((states_batchs, state_batch), axis=0)
                mcts_probs_batches = np.concatenate((mcts_probs_batches, mcts_probs_batch), axis=0)
                winner_batches = np.concatenate((winner_batches, winner_batch), axis=0)
    # Load the buffer with  Collected Data

    for i in range(states_batchs.shape[0]):
        replaybuffer.add_experience(states_batchs[i], mcts_probs_batches[i], winner_batches[i])

    return replaybuffer

# replaybuffer = load_collected_data_to_buffer(replaybuffer)

def get_pretrain_neural_network(replaybuffer, load_pretrain_model=False):
    ################### Phase III - Train the DNN model with collected data #######################
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not load_pretrain_model:
        pre_train_time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # modeldir = os.path.join(ROOT_DIR, 'models')
        # if not os.path.exists(modeldir):
        #     os.makedirs(modeldir)
        print('Training the Policy-Value Network with Human, Random, and Computer Player Experiences')
        pvnet = PolicyValueNet()
        # Retrieve data from buffer to train
        datasize_to_train = replaybuffer.size()
        if replaybuffer.size() > training_config.batch_size:
            states_batchs, mcts_probs_batches, winner_batches = replaybuffer.samp_data(datasize_to_train)
            hist = pvnet.train(states_batchs, mcts_probs_batches, winner_batches, training_config.pre_epochs, show=True)
            pretrain_model_directory = os.path.join(ROOT_DIR, 'models', 'model', 'pretrain_model',
                                                    f'pretrain_model_selfplay_{training_config.pre_epochs}_epoch_{training_config.n_playout}_simulations')
            pvnet.save_model_BaghChal(pretrain_model_directory,
                                      f'pretrain_model_selfplay_{time_string}.h5')
            df = pd.DataFrame(hist.history)
            pvnet.save_csv(pretrain_model_directory, f'pretrain_model_selfplay_{time_string}.csv', df)
            # pd.DataFrame.to_csv(df, f'pretrain_model_selfplay_{training_config.pre_epochs}epoch_{training_config.n_playout}simulations_history.csv')
            # pvnet.save_model_plot(pretrain_model_directory, f'pretrain_model_selfplay_{training_config.pre_epochs}'
            #                                                 f'epoch_{training_config.n_playout}simulations_{time_string}.png')
    else:
        model_directory = os.path.join(ROOT_DIR, 'models', 'model', 'pretrain_model',
                                       f'pretrain_model_selfplay_{training_config.pre_epochs}'
                                       f'_epoch_{training_config.n_playout}_simulations')

        try:
            if os.path.exists(model_directory):
                files = os.listdir(model_directory)
                temp_date_time = []
                for file in files:
                    if file.endswith(".h5"):
                        temp_date_time.append(file.split('.')[0].split('_')[-1])
                temp_date_time = sorted(temp_date_time, key=lambda row: datetime.strptime(row, "%Y-%m-%d-%H-%M-%S"),
                                        reverse=True)
                integer_map = map(int, temp_date_time)
                integer_list = list(integer_map)
                sorted_list = sorted(integer_list, reverse=True)
                latest_model_file = 'pretrain_model_selfplay_' + str(sorted_list[0]) + '.h5'
            else:
                print(f"No pretrain model exits for current configuration, pre_epochs {training_config.pre_epochs}, epoch "
                      f"{training_config.epochs}, simulations {training_config.n_playout}")
        except Exception as exp:
            print("Exception occured ", exp)
            raise exp

        pvnet = PolicyValueNet(os.path.join(model_directory, latest_model_file))
        print('Previously Trained Model is Loaded')

    return pvnet

# pvnet = get_pretrain_neural_network()

def run_self_play_with_monte_carlo_tree_search(replaybuffer, pvnet):
    ##################################################################################################
    print(f"Saving selfplay data in {os.path.join(ROOT_DIR, 'selfplay_data')}")
    datadir = os.path.join(ROOT_DIR, 'selfplay_data_sole')
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    ##################################################################################################

    actual_winner = []
    NumOfGames = training_config.num_of_games
    for gamenumber in range(NumOfGames):
        game = Game()
        print('SelfPlay, Game Number:', gamenumber + 1)
        pvnet.loss_train_op()
        pvnet_fn = pvnet.policy_value_fn
        player = MCTSPlayer(pvnet_fn, n_playout=training_config.n_playout, is_selfplay=1)
        game.board.reset()
        start_time = int(datetime.now().timestamp())
        [w, data] = game.start_self_play(player, show=False)
        actual_winner.append(w)
        end_time = int(datetime.now().timestamp())
        game_time = end_time - start_time
        print('Winner:', w, '---- Game Time (Secs):', game_time)
        data=[x for x in data]
        data=symmetry_board_moves(data)
        state_batch = [x[0] for x in data]
        mcts_probs_batch = [x[1] for x in data]
        winner_batch = [x[2] for x in data]
        state_batch = np.array(state_batch)
        mcts_probs_batch = np.array(mcts_probs_batch)
        winner_batch = np.array(winner_batch)
        for i in range(state_batch.shape[0]):
            replaybuffer.add_experience(state_batch[i], mcts_probs_batch[i], winner_batch[i])
        # Keeping the game data incase we need for future
        savedict = {'Game Duration':game_time, 'Game Number': gamenumber+1, 'state_batch': state_batch, 'mcts_probs_batch': mcts_probs_batch,
                    'winner_batch': winner_batch}
        time_string = str(int(datetime.now().timestamp()))
        filename = os.path.join(datadir, 'Game_' + str(gamenumber) + '_' +  time_string + '.mat')
        savemat(filename, savedict)
        # Updating model only after certain interval
        if (gamenumber + 1) % training_config.selfplay_dnn_update_interval == 0:
            # Retrieve data from buffer to train, We only want to use limited amount of data to update Network
            datasize_to_train = min(replaybuffer.size(), 10000)
            if replaybuffer.size() > training_config.batch_size:
                states_batchs, mcts_probs_batches, winner_batches = replaybuffer.samp_data(datasize_to_train)
                print('DNN is updating with recent game observations...........')
                hist = pvnet.train(states_batchs, mcts_probs_batches, winner_batches, training_config.epochs, show=False)
                self_play_model_directory = os.path.join(ROOT_DIR, 'models', 'model', 'self_play_model',
                                                         f'model_selfplay_{training_config.epochs}_epoch_{training_config.n_playout}_simulations_{gamenumber + 1}_gamenumber')

                pvnet.save_model_BaghChal(self_play_model_directory,
                                          f'model_selfplay_{time_string}.h5')

                df = pd.DataFrame(hist.history)
                pvnet.save_csv(self_play_model_directory,
                               f'model_selfplay_{time_string}.csv', df)
                # pvnet.save_model_plot(self_play_model_directory, f'model_selfplay_{training_config.epochs}'
                #                f'epoch_{training_config.n_playout}simulations_{gamenumber+1}gamenumber.png')

# def main():
#     replaybuffer = ReplayBuffer(training_config.buffer_size)
#
#     replaybuffer = load_collected_data_to_buffer(replaybuffer)
#
#     load_pretrain_model = False
#
#     if load_pretrain_model:
#         pvnet = get_pretrain_neural_network(replaybuffer, load_pretrain_model)
#
#     training_mode = 'load_pretrain_model'
#
#     if training_mode == 'self_play_only':
#         pvnet = PolicyValueNet()
#         run_self_play_with_monte_carlo_tree_search(pvnet)
#     else:
#         run_self_play_with_monte_carlo_tree_search(pvnet)
#
# if __name__ == '__main__':
#     main()