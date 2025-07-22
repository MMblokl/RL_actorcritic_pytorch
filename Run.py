import numpy as np
import gymnasium
import torch
import matplotlib.pyplot as plt
from SAC import SAC

# The device to use for pytorch:
# If cuda is not available it will be mega super slow 0_0 but will still work using the CPU.
global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function for smoothing a list of data points for a less chaotic plot.
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


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

            smoothmean = np.asarray(smooth(mean_rewards, 0.75))
            smoothstd = np.asarray(smooth(std_rewards, 0.75))

            # Determine the number of steps taken in the environment
            steps = np.arange(len(mean_rewards))

            # Plot the environment and create a std range of each method mean reward
            plt.plot(steps, smoothmean, label=method)
            plt.fill_between(steps,
                            smoothmean - smoothstd,
                            smoothmean + smoothstd,
                            alpha=0.1)

        plt.xlabel("Policy evaluation every 1000 steps")
        plt.ylabel("Summed average reward")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("learning_curve.png", dpi=300)


agent = SAC(evaluate=True)
agent.train()
# The evaluation learning reward log. Empty if evaluate=False
agent.reward_log
# Save policy weights
agent.save("./policy.pt")
# Load policy weights
agent.load("./policy.pt")
