# Assignment 2: REINFORCE and basic Actor-Critic Methods

Four python scripts are delivered for the assignment, which consist of the following:
1. Nnmodule.py
2. REINFORCE.py
3. AC.py
4. A2C.py

These scripts were created and run with the packages as detailed in Requirements.

### REINFORCE.py
This python file contains the structure of the REINFORCE algorithm used for the CartPole environment.
This script is used to call the REINFORCE algorithm as the specific agent used by Run.py.

### AC.py
This python file contains the structure of the Actor-Critic algorithm used for the CartPole environment.
This script is used to call the AC algorithm as the specific agent used by Run.py.

### A2C.py
This python file contains the structure of the Advantage Actor Critic algorithm used for the CartPole environment.
This script is used to call the A2C algorithm as the specific agent used by Run.py.

### Nnmodule.py
This python file contains the structure used for the Policy Neural Network and the Critic Neural Network.
The Policy Neural Network is used for REINFORCE.py, AC.py and A2C.py to maximize the policy for the CartPole environment using each algorithm method.
The Critic Neural Network is used for AC.py and A2C.py as the value network to evaluate the policy network actions.
This script is used to call the policy and critic network for the three algorithm methods.

### Run.py
This python file contains the main training loop that initializes the agent for the REINFORCE, AC and A2C algorithm and the specific hyperparameters used for each algoritm.
This file also contains the main training loop that runs each algortithm one after another with the CartPole environment and gives a learning curve of the Running rewards over the episodes.
This script is used to create the agent of each algorithm method for a CartPole environment until convergence over 5 repetitions.

## Running the code
The code is run by calling Run.py with the following command:
>python3 Run.py

This runs all three algorithm methods in the CartPole environment and gives a learning curve as a result showing the mean Running rewards for each method.

## Requirements
Python = 3.11
Pytorch
Matplotlib
Gymnasium

The requirements needed to run the experiment were obtained by using: 
>python -m pip freeze > requirements.txt.
This created the file containing the packages from the coding environment where the code for the experiment was created and used.

The entire pip environment can be downloaded using the requirements.txt file
>pip install -r  requirements.txt