from Nnmodule import Critic
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from REINFORCE import REINFORCE


# Current implementation uses 1 network that predicts BOTH the action probabilities and the critic value for that state.
# If needed, we can just move it to 2 networks like the base example.
class AC(REINFORCE):
    def __init__(self, gamma, alpha, beta, n_neurons, n_layers, n_step, env, device):
        # Inherit all values and objects from REINFORCE.
        super().__init__(
            gamma=gamma,
            alpha=alpha,
            n_neurons=n_neurons,
            n_layers=n_layers,
            n_step=n_step,
            env=env,
            device=device
            )
        
        # Save parameters
        self.beta = beta

        # Critic network, policy net is inherited from reinforce.
        self.criticnet = Critic(n_obs=self.n_obs, n_neurons=n_neurons, n_layers=n_layers, device=self.device)
        self.cri_optim = optim.Adam(self.criticnet.parameters(), lr=self.beta, amsgrad=True)


    def sample_action(self, observation):
        # Set the state as a tensor
        state = torch.tensor(observation, dtype=torch.float, device=self.device) #Turn env state into a tensor
        # Get the action for the current state according to the policy.
        probabilities = self.policynet(state)
        critic_val = self.criticnet(state)

        # Create a distribution and sample an action
        dist = Categorical(probabilities)
        action = dist.sample()
            
        # Save logarithm of the probabilities
        log_p = dist.log_prob(action)
        action = action.item()

        return log_p, action, critic_val
    
    
    def calc_q(self, returns, critic_vals):
        # Calculate the Q-value using the critic value [t+n] and the returns values.
        # Use no_grad to not let gradients mix
        with torch.no_grad():
            gammas = torch.tensor([self.gamma ** n for n in range(self.n_step + 1)], dtype=torch.float, device=self.device)
            qs = []
            for t in range(len(returns)):
                # Value to get the correct gamma from the "gammas" tensor
                # will always be gamma ** n
                # If the frame is shifted all the way to the right, the frame has become 1-step,
                # it will be gamma ** 1, while the critic return value will still be the last one in the list.
                n = np.min([self.n_step, len(returns) - t])

                # Q-value of the 
                Q = returns[t] + critic_vals[t:t+self.n_step][-1] * gammas[n]
                qs.append(Q)
            # Stack the q-values into a single tensor
            qs = torch.stack(qs)
            return qs

    def train(self, log_probs, rewards, critic_vals):
        # Set the returns and log probs as a single tensor for easy multiplication.
        returns = self.calculate_returns(rewards=rewards)

        # Stack all probs into a singular tensor.
        log_probs = torch.stack(log_probs)
        critic_vals = torch.stack(critic_vals)
        
        # Get the Q-return value, returns + Q[t+n] * gamma ** n
        q_values = self.calc_q(returns=returns, critic_vals=critic_vals)

        # Normalize the q_val;ues to avoid unstable learning, and also to possible avoid exponential weight increase.
        q_values = (q_values - q_values.mean()) / (q_values.std() + np.finfo(np.float32).eps.item())

        # Use the log probabily and the Q-value to calculate the loss.
        policy_loss = -(log_probs * q_values).sum()
        
        # Critic loss, the mean squared error of the critic values and the cumulative return values.
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


    # Important: This function is inherited by A2C as the syntax is the exact same
    def train_loop(self):
        # A loop for training the agent given a number of steps and a rate of testing the weights for plotting
        
        running_rew = 10
        while True:            
            # Get the first observation, or state by resetting the environment
            done, truncated = False, False
            obs, _ = self.env.reset()
            
            # Save each trace into a list
            log_probs, rewards, critic_vals = [], [], []
            
            while not (done or truncated):
                # Do the first action from the beginning
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