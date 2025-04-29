from Nnmodule import Policy, Critic
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from Memory import Memory


# Copy of SAC class


class TD3:
    def __init__(self, gamma, alpha, beta, n_neurons, n_layers, n_step, env, device):
        # Hyperparameter initialization for the class
        self.gamma = gamma
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.n_step = n_step
        self.beta = beta
        
        # The environment and device to store tensors are are initiated as class objects
        self.env = env
        self.device=device

        # Get the number of possible actions and observations for loading the Q network
        self.n_act = env.action_space.n
        state, _ = env.reset()
        self.n_obs = len(state)
        self.env.close()

        # Init the Policy network and the Q-networks
        self.policy = Policy(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=device)
        self.Q = Critic(n_obs=self.n_obs, n_neurons=n_neurons, n_layers=n_layers, device=self.device)
        self.Q2 = Critic(n_obs=self.n_obs, n_neurons=n_neurons, n_layers=n_layers, device=self.device)

        # Optimizer and loss to use
        self.pol_optim = optim.Adam(self.policynet.parameters(), lr=alpha, amsgrad=True)
        self.Q_optim = optim.Adam(self.Q.parameters(), lr=self.beta, amsgrad=True)
        self.Q2_optim = optim.Adam(self.Q2.parameters(), lr=self.beta, amsgrad=True)

        # Replay buffer
        self.mem = Memory(5e4)

        # Running_reward values for measuring the convergence to the optimum using a threshold to stop training
        self.running_rews = []


    def sample_action(self, observation):
        # Set the state as a tensor
        state = torch.tensor(observation, dtype=torch.float, device=self.device) #Turn env state into a tensor
        
        # This function needs to be changed to do target policy clipping
        

        return log_p, action, critic_val


    def train(self, log_probs, rewards, critic_vals):
        # Set the returns and log probs as a single tensor for easy multiplication.
        returns = self.calculate_returns(rewards=rewards)

        # Stack all probs into a singular tensor.
        log_probs = torch.stack(log_probs)
        critic_vals = torch.stack(critic_vals)
        
        # Get the Q-return value, returns + Q[t+n] * gamma ** n
        q_values = self.calc_q(returns=returns, critic_vals=critic_vals)

        # Normalize the q-values to avoid unstable learning, and also avoid exponential weight increase.
        q_values = (q_values - q_values.mean()) / (q_values.std() + np.finfo(np.float32).eps.item())

        # Advantage: Q(st, at) - V(st)
        advantage = q_values - critic_vals

        # Use the log probabily and the return value to calculate the loss.
        # Detach the tensor to not let the critic and policy network gradients mix
        policy_loss = -(log_probs * advantage.detach()).sum()
        
        # Critic loss, the mean squared error of the critic values and the q-values, aka advantage.
        critic_loss = torch.nn.functional.mse_loss(critic_vals, q_values)
        
        # Reset old gradient
        self.pol_optim.zero_grad()
        self.cri_optim.zero_grad()

        # Backpropagate the loss
        policy_loss.backward()
        critic_loss.backward()

        # Clip the gradient of the policy and critic net for even more stable training
        torch.nn.utils.clip_grad_norm_(self.policynet.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.criticnet.parameters(), max_norm=1.0)

        # Optimizer step
        self.pol_optim.step()
        self.cri_optim.step()


    def train_loop(self):
        # A loop for training the agent given a number of steps and a rate of testing the weights for plotting
        

        # 1: get obs
        # 2: Select action based on the policy, decide how.
        # 3: Get next_obs, rewards and done signal and save the transition to memory


        running_rew = 10
        while True:            
            # Get the first observation, or state by resetting the environment
            done, truncated = False, False
            obs, _ = self.env.reset()
            
            # Save each trace into a list
            log_probs, rewards, critic_vals = [], [], []
            
            while not (done or truncated):
                # Do the first action from the beginning
                # Get action using the clipping thing
                
                
                
                
                
                
                log_p, action, critic_val = self.sample_action(observation=obs)
                critic_val = critic_val[0] # Make critic_val a single value and not [value]
                
                # Recieve feedback from the environment using the chosen action
                next_obs, reward, done, truncated, _ = self.env.step(action)
                
                # Save the feedback for training.
                log_probs.append(log_p)
                rewards.append(reward)
                critic_vals.append(critic_val)

                # Set next state
                obs = next_obs

                # Train the network on the collected feedback from the environment when the environment exits.
                if (done or truncated):
                    self.train(log_probs=log_probs, rewards=rewards, critic_vals=critic_vals)

            # Close the environment as the agent is done for this current episode
            self.env.close()

            # Continue running reward and stop training if it passed threshold
            # This value is a measure of how long the rewards of the network have stayed consistent until it reaches a threshold.
            # Taken from the example REINFORCE algorithm in the PyTorch github.
            running_rew = 0.10 * np.sum(rewards) + (1 - 0.10) * running_rew
            
            # Save the current running reward for the learning curve.
            self.running_rews.append(running_rew)
            
            # Does the running smoothed reward reach the threshold? stop the training.
            # 10000 episode limit in case the network is REALLY unlucky to prevent an infinite loop.
            if (running_rew >= self.env.spec.reward_threshold) or (len(self.running_rews) >= 10000):
                break
        print(f"Training done! Reached a running reward value of {running_rew}, in {len(self.running_rews)} episodes")
