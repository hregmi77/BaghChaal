from typing import List

import numpy as np



class ReplayBuffer:
    def __init__(self, max_size: int):
        self.replay_buff: List = []
        self.max_size: int = max_size

    def add_experience(self, state_batch, mcts_prob_batch, winner_batch):
        if len(self.replay_buff) >= self.max_size:
            self.replay_buff.pop(0)

        self.replay_buff.append((state_batch, mcts_prob_batch, winner_batch))

    def samp_data(self, batch_size: int):
        batch_size_i: int = min(len(self.replay_buff), batch_size)
        batch_idxs = np.random.randint(len(self.replay_buff), size=batch_size_i)
        replay_buff_batch = [self.replay_buff[idx] for idx in batch_idxs]
        states_batch = [x[0] for x in replay_buff_batch]
        mcts_prob_batch = [x[1] for x in replay_buff_batch]
        winner_batch = [x[2] for x in replay_buff_batch]


        return states_batch, mcts_prob_batch, winner_batch

    def size(self):
        return len(self.replay_buff)
