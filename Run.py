
import numpy as np
import gymnasium
import torch
import matplotlib.pyplot as plt
from SAC import SAC

# The device to use for pytorch:
# If cuda is not available it will be mega super slow 0_0 but will still work using the CPU.
global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_curve(method_dict):
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
                            mean_rewards - std_rewards,
                            mean_rewards + std_rewards,
                            alpha=0.1)

        plt.xlabel("Policy evaluation every 1000 steps")
        plt.ylabel("Summed average reward")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("learning_curve.png", dpi=300)

def run():
    # Initialize the methods used and the reward dict that will contain the rewards for all methods over 5 repetitions
    hps = [0.3, 0.5, 1.0]
    reward_dict = {}

    # Hyperparameters
    gamma = 0.99 # Future reward discount factor
    alpha = 0.0005 # Learning rate for all networks
    tau = 0.995 # Update rate for the target networks
    batchsize = 128 # How many transitions are sampled from the buffer for calculating loss
    update_every = 2 # How many Q-net/Critic-net updates per policy update
    init_sample = 1000 # Initial sample of env steps before the real train loop starts.
    memsize = 1000000 # Max size of replay buffer
    n_neurons = 256 # Number of neurons in the critic and policy network in all hidden layers.
    n_layers = 2  # Number of hidden layers in the hidden layer block of the policy and critic network.
    max_steps = 500000 # Number of steps to run the alg each iteration.
    plotrate = 1000

    # Run 5 repetitions
    for coef in hps:

        # Initalize the rewards list
        reward_list = []

        # Run 5 repetitions for each method
        for run in range(5):
            agent = SAC(gamma=gamma,
                alpha = alpha,
                tau=tau,
                batchsize=batchsize,
                update_every = update_every,
                init_sample=init_sample,
                memsize=memsize,
                reg_coef = coef,
                max_steps = max_steps,
                plotrate = plotrate,
                n_neurons=n_neurons,
                n_layers=n_layers,
                env=gymnasium.make("CartPole-v1"),
                device=device
                )
            agent.train_loop()
            print(f"Finished iteration {run} for regularization coefficient {coef}")

            # Add the rewards for the current repetitions to the reward list
            reward_list.append(agent.reward_log)

        # Add the agent rewards to the reward list
        reward_dict[coef] = reward_list
 
    # Plot all methods in one plot.
    plot_curve(reward_dict)

if __name__ == "__main__":
    run()