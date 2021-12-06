from CollectObservations import generate_data
from policy_value_net import PolicyValueNet
from rl_utils import ReplayBuffer
from main_baghchal_all import load_collected_data_to_buffer
from main_baghchal_all import get_pretrain_neural_network
from main_baghchal_all import run_self_play_with_monte_carlo_tree_search
from evaluate_baghchal import evaluate_baghchal_game
from play_baghchal_game import play_baghchal
from config import TrainConfig

import os


training_config = TrainConfig()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def train_from_scratch():
    generate_data()
    replaybuffer = ReplayBuffer(training_config.buffer_size)
    replaybuffer = load_collected_data_to_buffer(replaybuffer)
    pvnet = get_pretrain_neural_network(replaybuffer)
    run_self_play_with_monte_carlo_tree_search(pvnet)

def train():
    replaybuffer = ReplayBuffer(training_config.buffer_size)
    replaybuffer = load_collected_data_to_buffer(replaybuffer)
    pvnet = get_pretrain_neural_network(replaybuffer)
    run_self_play_with_monte_carlo_tree_search(pvnet)

def train_self_play_model_only():
    replaybuffer = ReplayBuffer(training_config.buffer_size)
    pvnet = PolicyValueNet()
    print(replaybuffer.size())
    run_self_play_with_monte_carlo_tree_search(replaybuffer, pvnet)
    print(replaybuffer.size())

def train_self_play_with_pretrain_model():
    replaybuffer = ReplayBuffer(training_config.buffer_size)
    replaybuffer = load_collected_data_to_buffer(replaybuffer)
    pvnet = get_pretrain_neural_network(replaybuffer, load_pretrain_model=True)
    run_self_play_with_monte_carlo_tree_search(pvnet)

def main():
    while True:
        # Configure running options
        options = [
            "Generate Data",
            "Train both pretrain model and self play model",
            "Load pretrained model and train self play with pretrained model",
            "Generate Data, train pretrain model",
            "Train with self play model only",
            "Play Baghchal",
            "Evaluate model",
            "Exit",
        ]

        print()
        for i in range(len(options)):
            print(f"{i}. {options[i]}")

        choice = input("Enter a number to choose an action: ")
        valid_inputs = [str(i) for i in range(len(options))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)
        if choice == 0:
            generate_data()
        elif choice == 1:
            train()
        elif choice == 2:
            train_self_play_with_pretrain_model()
        elif choice == 3:
            train_from_scratch()
        elif choice == 4:
            train_self_play_model_only()
        elif choice == 5:
            play_baghchal()
        elif choice == 6:
            evaluate_baghchal_game()
        else:
            break

if __name__ == "__main__":
    main()