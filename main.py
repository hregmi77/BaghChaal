from game import Game
from player import MCTSPlayer,HumanPlayer
from policy_value_net import PolicyValueNet
from utils import symmetry_board_moves
game=Game()
goat=HumanPlayer()
bagh = HumanPlayer()
pvnet=PolicyValueNet()
pvnet_fn=pvnet.policy_value_fn
bagh=MCTSPlayer(pvnet_fn,n_playout=500)

# print(bagh.mcts.policy)
# exit(0)
# data=game.start_play(bagh,goat)
[w, data] = game.start_self_play(bagh)
data=[x for x in data]
data=symmetry_board_moves(data)
state_batch = [x[0] for x in data]
mcts_probs_batch = [x[1] for x in data]
winner_batch = [x[2] for x in data]
pvnet.train(state_batch,mcts_probs_batch,winner_batch,5)
pvnet.save_model("model.h5")

