from Nnmodule import Policy
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class REINFORCE:
    def __init__(self, gamma, alpha, n_neurons, n_layers, n_step, env, device):
        # Hyperparameter initialization for the class
        self.gamma = gamma
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.n_step = n_step
        
        # The environment and device to store tensors are are initiated as class objects
        self.env = env
        self.device=device

        # Get the number of possible actions and observations for loading the Q network
        self.n_act = env.action_space.n
        state, _ = env.reset()
        self.n_obs = len(state)
        self.env.close()

        # Init the Policy network
        self.policynet = Policy(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=device)

        # Optimizer and loss to use
        self.pol_optim = optim.Adam(self.policynet.parameters(), lr=alpha, amsgrad=True)
        
        # Running_reward values for measuring the convergence to the optimum using a threshold to stop training
        self.running_rews = []

    # This function is the exact same for AC and A2C.
    def calculate_returns(self, rewards):
        # Calculate the list of R values using n_step target
        # Calculate all gamma^n values beforehand.
        gammas = torch.tensor([self.gamma ** n for n in range(self.n_step)], dtype=torch.float, device=self.device)
        # Set the rewards as a tensor for easy arithmetic
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)

        returns = []
        for i in (range(len(rewards))):
            # Get the frame to calculate the return from using the self.n_step parameter
            frame = rewards[i:i+self.n_step]
            
            # Calculate return for the current t using the pre-computed gammas
            return_val = torch.sum(frame * gammas[:len(frame)])
            returns.append(return_val)
        
        # Stack the returns values into a single tensor.
        returns = torch.stack(returns)
        return returns

    def sample_action(self, observation):
        # Function to return the log_p value and the action from a state using sampling
        
        # Set the state as a tensor
        state = torch.tensor(observation, dtype=torch.float, device=self.device) #Turn env state into a tensor
        # Get the action for the current state according to the policy.
        probabilities = self.policynet(state)
        # Create a distribution and sample an action
        dist = Categorical(probabilities)
        # Sample action from the probabilities.
        action = dist.sample()
        
        # Save logarithm of the probabilities
        log_p = dist.log_prob(action)
        action = action.item()

        return log_p, action

    def train(self, log_probs, rewards):
        # Function to calculate loss and backpropagate it over the weights graph
        
        # Set the returns and log probs as a single tensor for easy multiplication.
        returns = self.calculate_returns(rewards=rewards)

        # Normalize the returns to avoid unstable learning, and to decrease variance.
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        # Stack all probs into a singular tensor.
        log_probs = torch.stack(log_probs)
        
        # Use the log probabily and the return value to calculate the loss.
        # This is the vanilla policy loss, expected reward * negative log
        # Compute the loss: -log_prob_t * return_t
        loss = -(log_probs * returns).sum()
        
        # Reset old gradient
        self.pol_optim.zero_grad()

        # Backpropagate the loss
        loss.backward()

        # Clip the gradient of the policy net to avoid unstable traininf
        torch.nn.utils.clip_grad_norm_(self.policynet.parameters(), max_norm=1.0)

        # Optimizer step
        self.pol_optim.step()

    def train_loop(self):
        # A loop for training the agent given a number of steps and a rate of testing the weights for plotting
        running_rew = 10
        while True:            
            # Get the first observation, or state by resetting the environment
            done, truncated = False, False
            obs, _ = self.env.reset()
            
            # Save each trace into a list
            log_probs, rewards = [], []
            
            while not (done or truncated):
                # Do the first action from the beginning
                log_p, action = self.sample_action(observation=obs)
                
                # Recieve feedback from the environment using the chosen action
                next_obs, reward, done, truncated, _ = self.env.step(action)
                
                # Save the feedback into the trace
                log_probs.append(log_p)
                rewards.append(reward)

                # Set next state
                obs = next_obs

                # Train if the episode is completed or when the current trace is equal to the batch size.
                if (done or truncated):
                    self.train(log_probs=log_probs, rewards=rewards)

            # Close the environment as the agent is done for this current episode
            self.env.close()

            # Continue running reward and stop training if it passed threshold
            # This value is a measure of how long the rewards of the network have stayed consistent until it reaches a threshold.
            # Taken from the example REINFORCE algorithm in the PyTorch github.
            running_rew = 0.10 * np.sum(rewards) + (0.90) * running_rew
            
            # Save the current running reward for the learning curve.
            self.running_rews.append(running_rew)
            
            # Agent converges if the running_rew is over the reward threshold.
            # 10000 max episodes if the agent is REALLY unlucky to avoid infinite loops.
            if (running_rew >= self.env.spec.reward_threshold) or (len(self.running_rews) >= 10000):
                break

        print(f"Training done! Reached a running reward value of {running_rew}, in {len(self.running_rews)} episodes")
