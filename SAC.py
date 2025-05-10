from Nnmodule import Policy, Critic
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from Memory import Memory, transition
import gymnasium


class SAC:
    def __init__(self, gamma, alpha, tau, batchsize, memsize, update_every, init_sample, reg_coef, max_steps, plotrate, n_neurons, n_layers, env, device):
        # Hyperparameter initialization for the algorithm class
        self.gamma = gamma
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.tau = tau
        self.memsize = memsize
        self.batchsize = batchsize
        self.update_every = update_every
        self.init_sample = init_sample
        self.regularization_coef = reg_coef
        self.max_steps = max_steps
        self.pt = plotrate
        
        # The environment and pytorch device are stored as local class variables.
        self.env = env
        self.device = device

        # Get the number of possible actions and observations for for intializing the critic and target nets
        self.n_act = env.action_space.n
        state, _ = env.reset()
        self.n_obs = len(state)
        self.env.close()

        # Init the Policy network, Critic networks and Target networks. The critic are the "Q1" and "Q2" net
        self.policy = Policy(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=device)
        self.Q1 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device) # Critic 1
        self.Q2 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device) # Critic 2
        self.T1 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device) # Target 1
        self.T2 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device) # Target 2

        # Initialize the target net parameters as a copy of the critic net parameters.
        self.T1.load_state_dict(self.Q1.state_dict().copy())
        self.T2.load_state_dict(self.Q2.state_dict().copy())

        # Optimizers for policy and critic networks. Target network is soft updated and thus does not need an optimizer.
        self.pol_optim = optim.Adam(self.policy.parameters(), lr=alpha, amsgrad=True)
        self.Q1_optim = optim.Adam(self.Q1.parameters(), lr=alpha, amsgrad=True)
        self.Q2_optim = optim.Adam(self.Q2.parameters(), lr=alpha, amsgrad=True)

        # Replay buffer/memory
        self.mem = Memory(memsize)

        # Running_reward values for measuring the convergence to the optimum using a threshold to stop training
        self.running_rews = []
        self.reward_log = []


    def sample_action(self, observation) -> int:
        # Randomly samples an action according to the action probabilities.
        # Set the state as a tensor
        state = torch.tensor(observation, dtype=torch.float, device=self.device)
        # Get the actions probs from the policy net.
        probabilities, _ = self.policy(state)
        # Action is sampled from the probabilaties with np.choice.
        action = np.random.choice(self.n_act, p=probabilities.cpu().numpy())

        return action


    def update_target(self) -> None:
        # Soft update the target networks
        q1_state = self.Q1.state_dict().copy()
        q2_state = self.Q2.state_dict().copy()
        t1_state = self.T1.state_dict().copy()
        t2_state = self.T2.state_dict().copy()

        # Take tau * target times the target weight + (1-tau) * critic weight
        for key in q1_state:
            t1_state[key] = t1_state[key]*self.tau + q1_state[key]*(1-self.tau)
        self.T1.load_state_dict(t1_state)
        for key in q2_state:
            t2_state[key] = t2_state[key]*self.tau + q2_state[key]*(1-self.tau)
        self.T2.load_state_dict(t2_state)


    def test_net(self) -> None:
        # Deterministic policy test for the policy. This function is called every n steps according to self.pt: plot_rate
        with torch.no_grad(): # No gradients are needed, so we use torch.no_grad()
            localenv = gymnasium.make("CartPole-v1") # Local environment that does not intersect with the one used for training.
            rewards = []
            done, truncated = False, False
            obs, _ = localenv.reset()
            while not (done or truncated):
                # Take the state as a tensor for policy usage
                state = torch.tensor(obs, device=self.device)
                probs, _ = self.policy(state)
                action = np.argmax(probs.cpu().numpy()) # Use a deterministic policy for testing
                next_obs, reward, done, truncated, _ = localenv.step(action)
                obs = next_obs
                # Take the current reward to save
                rewards.append(reward)
            # Close the local environment
            localenv.close()
        # Take the summed reward and save it to the reward log
        self.reward_log.append(np.sum(rewards))


    def train(self) -> None:
        # Sample batch of transitions from the replay buffer.
        batch = self.mem.sample(self.batchsize)
        # Seperate the (s, a, s', r, d) values from the batch
        batch = transition(*zip(*batch))
        
        # Take each group of variables from the batch and store them in tensors
        states = torch.tensor(np.array(batch.state), device=self.device) # Stack all states into an ndarray and transfer it to a tensor
        next_states = torch.tensor(np.array(batch.next_state), device=self.device)
        actions = torch.tensor(batch.action, device=self.device)
        rewards = torch.tensor(batch.reward, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float, device=self.device)
        
        # Do not calculate gradients for target value calculation to avoid gradients flowing incorrectly.
        with torch.no_grad():
            # Sample NEW actions using the next states for the target values
            next_action_probs, next_log_probs = self.policy(next_states)

            # Compute target values
            t1_vals = self.T1(next_states)
            t2_vals = self.T2(next_states)
            min_t_vals = torch.min(t1_vals, t2_vals)
            
            # Target value computation
            # H is estimated using the log_probs value
            # We use the full expectation, so from both actions, multiplied by the action probabilities.
            # This way, the algorithm learns from both actions.
            # Summed over both actions with the min_t_vals and the entropy.
            next_v = (next_action_probs * (min_t_vals - (self.alpha * next_log_probs))).sum(dim=1)
            y = rewards + (1-dones) * self.gamma * next_v # Target value y

        # Get Q-vals for both Q nets
        q1_vals = self.Q1(states).gather(1, actions.view(-1,1)).view(1,-1)[0]
        q2_vals = self.Q2(states).gather(1, actions.view(-1,1)).view(1,-1)[0]

        # Loss for the critic networks is MSE between the Q-values and the target value y.
        q1_loss = torch.nn.functional.mse_loss(q1_vals, y)
        q2_loss = torch.nn.functional.mse_loss(q2_vals, y)

        # Critic nets are updates BEFORE the policy loss is computed.
        # Reset old grads
        self.Q1_optim.zero_grad()
        self.Q2_optim.zero_grad()

        # Backpropagate the loss
        q1_loss.backward()
        q2_loss.backward()

        # Optimizer step
        self.Q1_optim.step()
        self.Q2_optim.step()

        # Is this train step a policy training step?
        if self.steps % self.update_every == 0:
            # Loss for the policy network
            act_probs, log_probs = self.policy(states)
            # Calculate the min(q1, q2) values without gradient to avoid gradients flowing incorrectly.
            with torch.no_grad():
                q1 = self.Q1(states)
                q2 = self.Q2(states)
                min_q_vals = torch.min(q1, q2)

            # Calculate the policy loss
            # We use the full expectation, so from both actions, multiplied by the action probabilities.
            # This way, the algorithm learns from both actions.
            # Summed over both actions with the min_t_vals and the entropy.
            pol_loss = (act_probs * (self.alpha * log_probs - min_q_vals)).sum(dim=1).mean()
        
            # Clear old gradient
            self.pol_optim.zero_grad()
            # Backprop loss
            pol_loss.backward()
            # Update network
            self.pol_optim.step()

        # Update target networks using soft update
        self.update_target()


    def train_loop(self):
        # A loop for training the agent given a number of steps and a rate of testing the weights for plotting
        # Init steps and running reward.
        self.steps = 0
        while self.steps < self.max_steps:
            # Get the first observation, or state by resetting the environment
            done, truncated = False, False
            obs, _ = self.env.reset()
            
            while not (done or truncated):
                # Sample action from the policy
                with torch.no_grad():
                    action = self.sample_action(observation=obs)
                
                # Recieve feedback from the environment using the chosen action
                next_obs, reward, done, truncated, _ = self.env.step(action)

                # Save transition to replay buffer
                self.mem.save(obs, action, next_obs, reward, (done or truncated))

                # Set next state
                obs = next_obs
                self.steps += 1

                # Train networks if the initial sampling is finished.
                if self.steps > self.init_sample:
                    self.train()
                
                # Is it time to generate a test data point?
                if self.steps % self.pt == 0:
                    self.test_net()
            
            # Close the environment as the agent is done for this current episode
            self.env.close()
