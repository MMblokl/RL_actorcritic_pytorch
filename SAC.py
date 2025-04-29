from Nnmodule import Policy, Value
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from Memory import Memory, transition


# VERY good source:
# https://spinningup.openai.com/en/latest/algorithms/sac.html#id12
# Currently a completely seperate A2C algorithm.
# Basically:
# The entropy of the variable that comes out of the policy is taken into account while training.
# Entropy can be evaluated for a specific variable, using the function or distribution it was derived from.
# This way, entropy H of variable x is computed from the distribution P.
# H(P) = Ex~P[-log(P(x))]

# The agent gets a bonus to the reward equal to:
# reward + sigma * H(pi(state))
# Here, sigma is the trade-off coeffcient, where an infinite-step n-step is used.
# Reward is the R-value for a state, while H is the entropy of the action chosen from the policy given that state.

# Therefore, the way the critic/value-function is trained also changes in accordance to this.
# The entropy bonus from each timestep for V(s) is thus added using sigma.

# The Q-value derived from the n-step critic-value and reward value is then also changed to take a discounted entopy into account.
# The bellman equation for Q is thus changed.

# With that, training value function V is done using the Q-values + sigma * H(Pi(.|s))

# Now, SAC works a bit differently again. TWO Q-funtions, or just 2 Q-values from different critic nets are learnt.
# This comed from TD3 where SAC is based off of.
# Honestly not too sure how that would work:


# BASE:
# Twin delayed DDPG, learns TWO Q-functions.:
# https://spinningup.openai.com/en/latest/algorithms/td3.html
# Two Q-functions are learnt. So actually this does not actually use critic functions.
# The policy is learned by simply doing a MAX on the Q_1.

# THe Q-functions are learned using "clipped double-Q learning", were both Q-functions use 1 target value, calculated using 
# which Q-function gives a smaller target value.
# y(r, s', d) = r + gamma*(1-d)*minQ_i(s', a'(s'))
# Both Q-functions are learned by doing regression towards this target: MSE loss
# Loss(Q_1) = E[(Q_1(s,a) - y(r, s', d))^2]
# Loss(Q_2) = E[(Q_2(s,a) - y(r, s', d))^2]

# y is our target value, derived from 1 of the Q-functions with the weaker Q-value, in order to not give bias to one Q-function.

# The policy is learned using:
# max Expected_r[Q_1(s,pi(s))]
# End at convergence.

# Then, the DELAY comes in that the Q-functions update normally while the policy
# Updates with a delay. The paper recommends 2 Q-net updates each pol-net update.





class SAC:
    def __init__(self, gamma, alpha, beta, udr, reg_coef, n_neurons, n_layers, n_step, env, device):
        # Hyperparameter initialization for the class
        self.gamma = gamma
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.n_step = n_step
        self.beta = beta
        self.updaterate = udr
        self.regularization_coef = reg_coef
        
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
        self.Q1 = Value(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device)
        self.Q2 = Value(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device)
        self.T1 = Value(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device)
        self.T2 = Value(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device)

        # Set target net parameters as the Q-net parameters
        self.T1.load_state_dict(self.Q1.state_dict().copy())
        self.T2.load_state_dict(self.Q2.state_dict().copy())

        # Optimizer and loss to use
        self.pol_optim = optim.Adam(self.policy.parameters(), lr=alpha, amsgrad=True)
        self.Q1_optim = optim.Adam(self.Q1.parameters(), lr=self.beta, amsgrad=True)
        self.Q2_optim = optim.Adam(self.Q2.parameters(), lr=self.beta, amsgrad=True)
        self.T1_optim = optim.Adam(self.T1.parameters(), lr=self.beta, amsgrad=True)
        self.T2_optim = optim.Adam(self.T2.parameters(), lr=self.beta, amsgrad=True)

        # Replay buffer
        self.mem = Memory(10000)

        # Running_reward values for measuring the convergence to the optimum using a threshold to stop training
        self.running_rews = []


    def sample_action(self, observation):
        # Set the state as a tensor
        state = torch.tensor(observation, dtype=torch.float, device=self.device) #Turn env state into a tensor
        # Get the action for the current state according to the policy.
        probabilities = self.policy(state)

        # Create a distribution and sample an action
        dist = Categorical(probabilities)
        action = dist.sample()
            
        # Save logarithm of the probabilities
        log_p = dist.log_prob(action)
        action = action.item()

        return action


    def target_sample(self, observations):
        with torch.no_grad():
            # Get the probabilaty dists
            probs = self.policy(observations)
            dists = [Categorical(p) for p in probs]

            # Get actions
            actions = [dist.sample() for dist in dists]
            
            # Get log values for target computation
            log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]

            # Set the log_probs and actions to tensors
            actions = torch.tensor([np.int64(i) for i in actions])
            log_probs = torch.tensor([np.float64(i) for i in log_probs])

            return actions, log_probs
    
    def reparam_sample(self, observations)
        probs = 


    def train(self):
        # Sample batch from replaymem
        batch = self.mem.sample(self.updaterate)
        batch = transition(*zip(*batch))
        
        # Get the values into tensors for math
        states = torch.tensor(np.array(batch.state), device=self.device) # Stack all states into an ndarray and transfer it to a tensor
        next_states = torch.tensor(np.array(batch.next_state), device=self.device)
        actions = torch.tensor(batch.action, device=self.device)
        rewards = torch.tensor(batch.reward, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float, device=self.device)
        
        # Sample NEW actions using the states for the target functions
        t_act, t_probs = self.target_sample(states)

        # Compute target values
        with torch.no_grad():
            t1_vals = self.T1(next_states).gather(1, t_act.view(-1,1)).view(1,-1)[0]
            t2_vals = self.T2(next_states).gather(1, t_act.view(-1,1)).view(1,-1)[0]

        # Target value
        y = rewards + self.gamma*(1-dones)*(torch.min(t1_vals, t2_vals) - self.regularization_coef * t_probs)

        # Get Q-vals for both Q nets
        q1_vals = self.Q1(states).gather(1, actions.view(-1,1)).view(1,-1)[0]
        q2_vals = self.Q2(states).gather(1, actions.view(-1,1)).view(1,-1)[0]

        # Loss for the policy network

        # SOMETHING REPARAMETRIZATION TRICK????????


        # MSEloss between the target y and the output q values
        q1_loss = torch.nn.functional.mse_loss(q1_vals, y)
        q2_loss = torch.nn.functional.mse_loss(q2_vals, y)
        breakpoint()

        # Reset old grads
        self.Q1_optim.zero_grad()
        self.Q2_optim.zero_grad()

        # Backpropagate the loss
        q1_loss.backward()
        q2_loss.backward()

        # Optimizer step
        self.Q1_optim.step()
        self.Q2_optim.step()


    def train_loop(self):
        # A loop for training the agent given a number of steps and a rate of testing the weights for plotting
        steps = 0
        running_rew = 10
        while True:            
            # Get the first observation, or state by resetting the environment
            done, truncated = False, False
            obs, _ = self.env.reset()

            while not (done or truncated):

                # Sample action from the policy
                action = self.sample_action(observation=obs)
                
                # Recieve feedback from the environment using the chosen action
                next_obs, reward, done, truncated, _ = self.env.step(action)
                
                # Save transition to replay buffer
                self.mem.save(obs, action, next_obs, reward, done)

                # Set next state
                obs = next_obs
                steps += 1

                # Train the networks
                if steps % self.updaterate == 0:
                    self.train()

            # Close the environment as the agent is done for this current episode
            self.env.close()

            

            # Continue running reward and stop training if it passed threshold
            # This value is a measure of how long the rewards of the network have stayed consistent until it reaches a threshold.
            # Taken from the example REINFORCE algorithm in the PyTorch github.
            #running_rew = 0.10 * np.sum(rewards) + (1 - 0.10) * running_rew
            
            # Save the current running reward for the learning curve.
            self.running_rews.append(running_rew)
            
            # Does the running smoothed reward reach the threshold? stop the training.
            # 10000 episode limit in case the network is REALLY unlucky to prevent an infinite loop.
            if (running_rew >= self.env.spec.reward_threshold) or (len(self.running_rews) >= 10000):
                break
        print(f"Training done! Reached a running reward value of {running_rew}, in {len(self.running_rews)} episodes")
