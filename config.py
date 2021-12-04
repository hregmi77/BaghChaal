class ModelConfig:
    def __init__(self):
        self.cnn_first_filter_num=128
        self.cnn_filter_num = 64
        self.cnn_first_filter_size = 3
        self.cnn_filter_size = 2
        self.resnet_N = 5
        self.l2_reg = 1e-4
        self.value_dense_size = 32

class TrainConfig:
    def __init__(self):
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 0.8  # the temperature param
        self.n_playout = 50  # num of simulations for each move
        self.c_puct = 0.001
        self.buffer_size = 20000
        self.batch_size = 32  # mini-batch size for training

        self.play_batch_size = 1 # how many games to play
        self.epochs = 7  # num of train_steps for each update
        self.pre_epochs = 40
        self.ld=0.8
        self.elo_baserating = 1500
        self.elo_k_factor = 32
        self.num_of_games = 100
        self.num_of_random_minmax_games = 100
        self.num_of_minmax_minmax_games = 50
        self.allow_pretrain = 1
        self.selfplay_dnn_update_interval = 1

        #c_puct=0.0001 ld =0.8
class TrainGreedyConfig:
    def __init__(self):
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 0.8  # the temperature param
        self.n_playout = 800  # num of simulations for each move
        self.c_puct = 1
        self.buffer_size = 10000
        self.batch_size = 1  # mini-batch size for training

        self.play_batch_size = 1 # how many games to play
        self.epochs = 7  # num of train_steps for each update
        self.ld=0.8

        #c_puct=0.0001 ld =0.8
