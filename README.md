# Actor critic approaches using PyTorch.
This repository contains 3 Actor-Critic reinforcement learning algorithms created using PyTorch using a class-based agent:
- Actor Critic (AC)
- Advantage Actor Critic (A2C)
- Soft Actor Critic (SAC)

# Plotting learning curve.
The example Run.py script contains a plotting function using matplotlib that plots the evaluation returns if evaluate was set to true.
Note that the evaluation is set up to evaluate the current policy using 5 new environments after every 1000 training steps.
This means that variation from different initializations are not accounted for. To do this, simply repeat the training n number of times and take the average to account for this variation.