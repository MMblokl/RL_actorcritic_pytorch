# Assignment 3: Advanced Actor Critic - PPO and/or SAC

Four python scripts are delivered for the assignment, which consist of the following:
1. Nnmodule.py
2. SAC.py
3. Run.py

These scripts were created and run with the packages as detailed in Requirements.

### SAC.py
This file contains the entire SAC algorithm.

### Nnmodule.py
Python file containing the Critic and Policy network used in the SAC algorithm.

### Run.py
This file contains the skeleton used to run SAC 5 times, the hyperparameters, and the plotting function for the episode return plot.

## Running the code
The code is run by calling Run.py with the following command:
>python3 Run.py

## Requirements
Python = 3.11
Pytorch
Matplotlib
Gymnasium
Pytorch

The requirements needed to run the experiment were obtained by using: 
>python -m pip freeze > requirements.txt.
This created the file containing the packages from the coding environment where the code for the experiment was created and used.

The entire pip environment can be downloaded using the requirements.txt file
>pip install -r  requirements.txt

## Note:
There might be issues with the requirements.txt file as the pip freeze commands tends to link back to local package cache files at times.
If this occurs, just install Gymnasium, Pytorch, Matplotlib and numpy seperately, as long as the CUDA version of pytorch is used, this should just work.