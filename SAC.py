from Nnmodule import Policy, Critic
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from Memory import Memory, transition
import gymnasium


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
    def __init__(self, gamma, alpha, tau, batchsize, memsize, update_every, init_sample, reg_coef, max_eps, n_neurons, n_layers, env, device):
        # Hyperparameter initialization for the class
        self.gamma = gamma
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.tau = tau
        self.memsize = memsize
        self.batchsize = batchsize
        self.update_every = update_every
        self.init_sample = init_sample
        self.regularization_coef = reg_coef
        self.max_eps = max_eps
        
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
        self.Q1 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device)
        self.Q2 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device)
        self.T1 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device)
        self.T2 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device)

        # Set target net parameters as the Q-net parameters
        self.T1.load_state_dict(self.Q1.state_dict().copy())
        self.T2.load_state_dict(self.Q2.state_dict().copy())

        # Optimizer and loss to use
        self.pol_optim = optim.Adam(self.policy.parameters(), lr=alpha, amsgrad=True)
        self.Q1_optim = optim.Adam(self.Q1.parameters(), lr=alpha, amsgrad=True)
        self.Q2_optim = optim.Adam(self.Q2.parameters(), lr=alpha, amsgrad=True)

        # Replay buffer
        self.mem = Memory(memsize)

        # Running_reward values for measuring the convergence to the optimum using a threshold to stop training
        self.running_rews = []


    def sample_action(self, observation) -> int:
        # Set the state as a tensor
        state = torch.tensor(observation, dtype=torch.float, device=self.device) #Turn env state into a tensor
        # Get the action for the current state according to the policy.
        probabilities, _ = self.policy(state)

        # Action is sampled from the probabilaties
        action = np.random.choice(self.n_act, p=probabilities.numpy())

        return action


    def update_target(self) -> None:
        # Soft update the target networks
        q1_state = self.Q1.state_dict().copy()
        q2_state = self.Q2.state_dict().copy()
        t1_state = self.T1.state_dict().copy()
        t2_state = self.T2.state_dict().copy()

        for key in q1_state:
            t1_state[key] = t1_state[key]*self.tau + q1_state[key]*(1-self.tau)
        self.T1.load_state_dict(t1_state)
        for key in q2_state:
            t2_state[key] = t2_state[key]*self.tau + q2_state[key]*(1-self.tau)
        self.T2.load_state_dict(t2_state)


    def train(self) -> None:
        # Sample batch from replaymemory
        batch = self.mem.sample(self.batchsize)
        batch = transition(*zip(*batch))
        
        # Get the values into tensors for math
        states = torch.tensor(np.array(batch.state), device=self.device) # Stack all states into an ndarray and transfer it to a tensor
        next_states = torch.tensor(np.array(batch.next_state), device=self.device)
        actions = torch.tensor(batch.action, device=self.device)
        rewards = torch.tensor(batch.reward, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            # Sample NEW actions using the next states for the target values
            next_action_probs, next_log_probs = self.policy(states)


            # Compute target values
            t1_vals = self.T1(next_states)
            t2_vals = self.T2(next_states)
            min_t_vals = torch.min(t1_vals, t2_vals)

            # Next V-value
            # H is estimated using the log_probs value
            next_v = (next_action_probs * (min_t_vals - (self.alpha * next_log_probs))).sum(dim=1)

            # Target value
            y = rewards + (1-dones) * self.gamma * next_v

        # Get Q-vals for both Q nets
        q1_vals = self.Q1(states).gather(1, actions.view(-1,1)).view(1,-1)[0]
        q2_vals = self.Q2(states).gather(1, actions.view(-1,1)).view(1,-1)[0]

        # MSEloss between the target y and the output q values
        q1_loss = torch.nn.functional.mse_loss(q1_vals, y)
        q2_loss = torch.nn.functional.mse_loss(q2_vals, y)

        # Is it time to update the policy?
        if self.steps % self.update_every == 0:
            # Loss for the policy network
            act_probs, log_probs = self.policy(states)
            with torch.no_grad():
                q1 = self.Q1(states)
                q2 = self.Q2(states)
                min_q_vals = torch.min(q1, q2)

            # Calculate the policy loss
            pol_loss = (act_probs * (self.alpha * log_probs - min_q_vals)).sum(dim=1).mean()
        
            # Clear old gradient
            self.pol_optim.zero_grad()
            # Backprop loss
            pol_loss.backward()
            # Update network
            self.pol_optim.step()

        # Reset old grads
        self.Q1_optim.zero_grad()
        self.Q2_optim.zero_grad()

        # Backpropagate the loss
        q1_loss.backward()
        q2_loss.backward()

        # Optimizer step
        self.Q1_optim.step()
        self.Q2_optim.step()

        # Update target networks using soft update
        self.update_target()


    def train_loop(self):
        # A loop for training the agent given a number of steps and a rate of testing the weights for plotting
        self.steps = 0
        running_rew = 10
        while True:
            # Get the first observation, or state by resetting the environment
            done, truncated = False, False
            obs, _ = self.env.reset()
            
            rewards = []
            while not (done or truncated):
                # Sample action from the policy
                with torch.no_grad():
                    action = self.sample_action(observation=obs)
                
                # Recieve feedback from the environment using the chosen action
                next_obs, reward, done, truncated, _ = self.env.step(action)
                
                # Save current ep cumulative reward.
                rewards.append(reward)

                # Save transition to replay buffer
                self.mem.save(obs, action, next_obs, reward, (done or truncated))

                # Set next state
                obs = next_obs
                self.steps += 1

                # Train networks
                if self.steps > self.init_sample:
                    self.train()
            
            # Close the environment as the agent is done for this current episode
            self.env.close()

            # Update running reward
            running_rew = 0.10 * np.sum(rewards) + (1 - 0.10) * running_rew
            self.running_rews.append(running_rew)

            print(running_rew, np.sum(rewards))
            # Does the running smoothed reward reach the threshold? stop the training.
            # 10000 episode limit in case the network is REALLY unlucky to prevent an infinite loop.
            if (running_rew >= self.env.spec.reward_threshold) or (len(self.running_rews) >= self.max_eps):
                break
        print(f"Training done! Reached a running reward value of {running_rew}, in {len(self.running_rews)} episodes")
