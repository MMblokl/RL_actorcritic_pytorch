from Nnmodule import Critic, Policy
import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym


# Current implementation uses 1 network that predicts BOTH the action probabilities and the critic value for that state.
# If needed, we can just move it to 2 networks like the base example.
class AC:
    def __init__(self, 
                 gamma = 0.99,              # Discount factor for future reward
                 alpha = 1e-4,              # Policy learning rate
                 beta = 1e-3,               # Critic net learning rate
                 n_neurons = 256,           # Number of network neurons each layer
                 n_layers = 2,              # Number of hidden layers for both networks
                 n_step = 500,              # The n-step factor for calculating n-step reward
                 max_steps = 1000000,       # Max training steps
                 plot_rate = 1000,          # Number of steps to pass before a datapoint is generated
                 n_reps = 5,                # Number of evaluation repetitions to use (number of eval envs)
                 evaluate = True,           # Whether to use evaluation yes/no
                 envname = "CartPole-v1",   # Environment name
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 ):
        # Save parameters
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.n_step = n_step
        self.max_steps = max_steps
        self.envname = envname
        self.n_reps = n_reps
        self.pt = plot_rate
        self.evaluate = evaluate
        
        # The learning environment
        self.env = gym.make(envname)
        self.device = device
        self.n_act = self.env.action_space.n
        state, _ = self.env.reset()
        self.n_obs = len(state)
        self.env.close()

        # Init the Policy network and critic
        self.policy = Policy(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=device)
        self.pol_optim = optim.Adam(self.policy.parameters(), lr=alpha, amsgrad=True)
        self.critic = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device)
        self.cri_optim = optim.Adam(self.critic.parameters(), lr=self.beta, amsgrad=True)

        self.reward_log = []


    def sample_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float, device=self.device) #Turn env state into a tensor
        probabilities, log_p = self.policy(state)
        critic_val = self.critic(state)
        action = np.random.choice(self.n_act, p=probabilities.detach().cpu().numpy())
        return log_p[action], action, critic_val


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
            return_val = torch.sum(frame * gammas[:len(frame)])
            returns.append(return_val)
        
        # Stack the returns values into a single tensor.
        returns = torch.stack(returns)
        return returns


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

                Q = returns[t] + critic_vals[t:t+self.n_step][-1] * gammas[n] # Q of current selected step
                qs.append(Q)
            # Stack the q-values into a single tensor
            qs = torch.stack(qs)
            return qs


    def eval(self) -> None:
        with torch.no_grad(): # No gradient
            envs = gym.make_vec(self.envname, render_mode=None, num_envs=self.n_reps, vectorization_mode="async")
            saved_rews = []
            dones, truncateds = np.zeros(self.n_reps, dtype=bool), np.zeros(self.n_reps, dtype=bool)
            observations, _ = envs.reset()
            ep_end = np.zeros(self.n_reps) # Store the number of steps before termination
            steps = 0
            while np.any(ep_end == 0.): # Until all envs finish
                states = torch.tensor(observations, device=self.device)
                probs, _ = self.policy(states)
                actions = np.argmax(probs.cpu().numpy(), axis=1)
                next_obs, rewards, dones, truncateds, _ = envs.step(actions)
                observations = next_obs
                saved_rews.append(rewards)
                terminations = np.logical_or(dones, truncateds)
                steps += 1
                
                # Check whether to update ep_end
                if np.any(terminations):
                    loc = np.where(terminations)[0]
                    check = (ep_end[loc] == 0.)
                    ep_end[loc[check]] = steps
            saved_rews = np.array(saved_rews)
            #Sum the rewards using a mask method
            row_indices = np.arange(saved_rews.shape[0]).reshape(-1, 1)
            col_limits = ep_end.reshape(1, -1)
            mask = row_indices < col_limits
            final_rewards = np.sum(saved_rews * mask, axis=0)
            self.reward_log.append(final_rewards)
            envs.close()


    def train_ep(self, log_probs, rewards, critic_vals):
        # Set the returns and log probs as a single tensor for easy multiplication.
        returns = self.calculate_returns(rewards=rewards)

        # Stack all probs into a singular tensor.
        log_probs = torch.stack(log_probs)
        critic_vals = torch.stack(critic_vals)
        
        # Get the Q-return value, returns + Q[t+n] * gamma ** n
        q_values = self.calc_q(returns=returns, critic_vals=critic_vals)
        q_values = (q_values - q_values.mean()) / (q_values.std() + np.finfo(np.float32).eps.item()) # normalize q for unstable learning avoidance
        policy_loss = -(log_probs * q_values).sum()
        critic_loss = torch.nn.functional.mse_loss(critic_vals, q_values)
        self.pol_optim.zero_grad()
        self.cri_optim.zero_grad()
        policy_loss.backward()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.pol_optim.step()
        self.cri_optim.step()


    # Important: This function is inherited by A2C as the syntax is the exact same
    def train(self):
        self.steps = 0
        while self.steps < self.max_steps:
            done, truncated = False, False
            obs, _ = self.env.reset()
            
            # Save each trace into a list
            log_probs, rewards, critic_vals = [], [], []
            
            while not (done or truncated):
                log_p, action, critic_val = self.sample_action(observation=obs)
                critic_val = critic_val[0] # Make critic_val a single value and not [value]
                next_obs, reward, done, truncated, _ = self.env.step(action)
                
                # Save the feedback for training.
                log_probs.append(log_p)
                rewards.append(reward)
                critic_vals.append(critic_val)
                obs = next_obs
                
                # Train policy on entire return.
                if (done or truncated):
                    self.train_ep(log_probs=log_probs, rewards=rewards, critic_vals=critic_vals)
                
                # Evaluate 
                if (self.steps % self.pt == 0) and self.evaluate:
                    self.eval()

                self.steps += 1
            self.env.close()


    def save(self, filename):
        # Save policy parameters.
        torch.save(self.policy.state_dict(), filename)


    def load(self, filename):
        # Load policy parameters
        self.policy.load_state_dict(torch.load(filename))