BaghChal with MCTS

Here, we have three main files CollectObservations.py which collects data for pretraining neural network, main_baghchal_all.py which
is the core file of the project that collects data, pretrains model and run self play.

Then the other file is evlauate_baghchal.py with which we can evaluate the model that has been trained.
With play_baghchal_game.py we can play trained mcts with untrained minmax, untrained random player and untrained mcts.

The directory structure afer CollectObservations.py will be

        ---------->BaghChaalProject
                 ------------>BaghAsMinMax_GoatAsMinMax
                 ------------>BaghAsMinMax_GoatAsRandom
                 ------------>GoatAsMinMax_BaghAsRandom

We also have self curated humanplayerdata. The directory structure fo human player data will be 
        ------------>BaghChaalProject
                -------------->humanplayerdata

The models that are trained with collected data and self play data are saved in models directory
        ---------->BaghChaalProject
                 ----->models
                    ----->model
                        ----->pretrain_model # for model trained with collected data
                        ----->self_play_model # for model trained with mcts self play with or without pretrained model
                        
The data trained with selfplay model is saved in folder selfplay_data
The data trained with selfplay model is saved in folder selfplay_data_only

We can run the code by running main.py

Where we can choose between options
        [
            "Generate Data",
            "Train both pretrain model and self play model",
            "Load pretrained model and train self play with pretrained model",
            "Generate Data, train pretrain model",
            "Train with self play model only",
            "Play Baghchal",
            "Evaluate model",
            "Exit",
        ]
        
        The first option is used to generate data only.
        The second option is used to train both pretrain model with collected data and self play model.
        The third option is used to load previously trained pretrain model and self play model.
        The fourth option is used to generate data and train pretrain and self play model. This is if we want to start all the baghchal data collection and training process from scratch.
        The fifth option is used to train self play mcts model without pretrain model and colledted data.
        The sixth option is used to play baghchal using information from mcts network.
        The seventh option is used to evaluate result of previously trained model.
        The eigth option is used to get out of the main loop.

Here, since we have trained collected and trained data we can

run python main.py and select option 5.

But for that we will have to download data from ggogle drive and place it in BaghChal folder
