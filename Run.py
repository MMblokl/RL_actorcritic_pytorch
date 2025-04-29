
import numpy as np
import gymnasium
import torch
import matplotlib.pyplot as plt
from SAC import SAC

# The device to use for pytorch:
# If cuda is not available it will be mega super slow 0_0 but will still work using the CPU.
global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_curve(method_dict, stdrange: int, ):
        """
        Function for plotting the summed reward. May have issues
        if the total number of rewards is lower than 100.
        params:
        rewards: Tuple, tuple of summed reward over episode
        method: string, string to give the plot a proper title
        finished: boolean, bool to signify whether to save the plot y/n
        """

        plt.figure(figsize=(10, 6))

        for method, rewards_list in method_dict.items():
            # Determine the length of the the largest number of rewards for the reppetions
            max_len = max(len(reward_rep) for reward_rep in rewards_list)

            # Pad the rewards of each reppetion with nan values
            padded_r = np.array([reward_rep + [np.nan] * (max_len - len(reward_rep)) for reward_rep in rewards_list])

            # Calculate the mean and std without the nan values
            mean_rewards = np.nanmean(padded_r, axis=0)
            std_rewards = np.nanstd(padded_r, axis=0)

            # Determine the number of steps taken in the environment
            steps = np.arange(len(mean_rewards))

            # Plot the environment and create a std range of each method mean reward
            plt.plot(steps, mean_rewards, label=method)
            plt.fill_between(steps,
                            mean_rewards - stdrange,
                            mean_rewards + stdrange,
                            alpha=0.2)

        plt.xlabel("Gradient steps")
        plt.ylabel("Running reward")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("learning_curve.png")

def run():
    # Initialize the methods used and the reward dict that will contain the rewards for all methods over 5 repetitions
    #methods = ["REINFORCE", "AC", "A2C"]
    methods = ["A2C"]
    reward_dict = {}

    # Hyperparameters
    gamma = 0.99 # Future reward discount factor
    alpha = 1e-4 # Learning rate for the policy network
    beta = 1e-3 # Learning rate for critic network
    n_neurons = 256 # Number of neurons in the critic and policy network in all hidden layers.
    n_layers = 2  # Number of hidden layers in the hidden layer block of the policy and critic network.

    # Run 5 repetitions
    for method in methods:

        # Initalize the rewards list
        reward_list = []

        # Run 5 repetitions for each method
        for run in range(5):
            agent = SAC(gamma=gamma,
                alpha = alpha,
                beta = beta,
                udr=100,
                n_neurons=n_neurons,
                n_layers=n_layers,
                n_step = 200,
                env=gymnasium.make("CartPole-v1"),
                device=device
                )
            agent.train_loop()

            # Add the rewards for the current repetitions to the reward list
            reward_list.append(agent.running_rews)

        # Add the agent rewards to the reward list
        reward_dict[method] = reward_list
 
    # Plot all methods
    plot_curve(reward_dict, stdrange=10)

if __name__ == "__main__":
    run()