from Nnmodule import Policy, Critic
import torch
import torch.optim as optim
import numpy as np
from Memory import Memory, transition
import gymnasium as gym

#TODO:
#3. Possible improvements: Change the way naming is done, and use a lot less comments, make it just make sense
#4. Minimize as many functions as possible.
#5. Make plots an opt-in feature, and add saving the policy parameters and loading


class SAC:
    def __init__(self, 
                 gamma=0.99,            # Future reward discount factor
                 alpha=0.0005,          # Learning rate for all networks
                 tau=0.995,             # Update rate for the target networks
                 batchsize=128,         # How many transitions are sampled from the buffer for calculating loss
                 memsize=1000000,       # Max size of replay buffer
                 update_every=2,        # How many Q-net/Critic-net updates per policy update
                 init_sample=1000,      # Initial sample of env steps before the real train loop starts.
                 reg_coef=0.3,          # The regularization coefficient for entropy maximization
                 max_steps=1000000,     # Number of steps to run the alg each iteration.
                 plotrate=1000,         # Rate at which test datapoints are generated. Can be opted out if no evaluation is needed
                 n_neurons=256,         # Number of neurons in the critic and policy network in all hidden layers.
                 n_layers=2,            # Number of hidden layers in the hidden layer block of the policy and critic network.
                 evaluate=False,         # Whether to generate evaluation data points for plotting learning curve.
                 n_reps=5,              # The number of repetitions for the evaluation.
                 envname="CartPole-v1", # The environment name
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') # PyTorch device to use. CPU by default if CUDA does not work.
                 ):
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
        self.envname = envname
        self.evaluate = evaluate
        self.n_reps = n_reps
        
        # Initial parameter collection
        self.env = gym.make(envname)
        self.device = device
        self.n_act = self.env.action_space.n
        state, _ = self.env.reset()
        self.n_obs = len(state)
        self.env.close()

        # Init the Policy network, Critic networks and Target networks.
        self.policy = Policy(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=device)
        self.C1 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device) # Critic 1
        self.C2 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device) # Critic 2
        self.T1 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device) # Target 1
        self.T2 = Critic(n_obs=self.n_obs, n_act=self.n_act, n_neurons=n_neurons, n_layers=n_layers, device=self.device) # Target 2
        self.T1.load_state_dict(self.C1.state_dict().copy())
        self.T2.load_state_dict(self.C2.state_dict().copy())
        self.pol_optim = optim.Adam(self.policy.parameters(), lr=alpha, amsgrad=True)
        self.C1_optim = optim.Adam(self.C1.parameters(), lr=alpha, amsgrad=True)
        self.C2_optim = optim.Adam(self.C2.parameters(), lr=alpha, amsgrad=True)

        # Replay buffer/memory
        self.mem = Memory(memsize)
        self.reward_log = []


    def sample_action(self, observation) -> int:
        # Random action sampling from policy
        state = torch.tensor(observation, dtype=torch.float, device=self.device)
        probabilities, _ = self.policy(state)
        action = np.random.choice(self.n_act, p=probabilities.cpu().numpy())
        return action


    def update_target(self) -> None:
        # Soft update the target networks
        q1_state = self.C1.state_dict().copy()
        q2_state = self.C2.state_dict().copy()
        t1_state = self.T1.state_dict().copy()
        t2_state = self.T2.state_dict().copy()

        # Take tau * target times the target weight + (1-tau) * critic weight
        for key in q1_state:
            t1_state[key] = t1_state[key]*self.tau + q1_state[key]*(1-self.tau)
        self.T1.load_state_dict(t1_state)
        for key in q2_state:
            t2_state[key] = t2_state[key]*self.tau + q2_state[key]*(1-self.tau)
        self.T2.load_state_dict(t2_state)


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


    def train_batch(self) -> None:
        # Train the networks on a batch
        batch = self.mem.sample(self.batchsize)
        batch = transition(*zip(*batch))
        states = torch.tensor(np.array(batch.state), device=self.device)
        next_states = torch.tensor(np.array(batch.next_state), device=self.device)
        actions = torch.tensor(batch.action, device=self.device)
        rewards = torch.tensor(batch.reward, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            # Sample NEW actions using the next states for the target values
            next_action_probs, next_log_probs = self.policy(next_states)
            t1_vals = self.T1(next_states)
            t2_vals = self.T2(next_states)
            min_t_vals = torch.min(t1_vals, t2_vals)
            next_v = (next_action_probs * (min_t_vals - (self.alpha * next_log_probs))).sum(dim=1)
            y = rewards + (1-dones) * self.gamma * next_v # Target value y

        # Q_values and loss 
        q1_vals = self.C1(states).gather(1, actions.view(-1,1)).view(1,-1)[0]
        q2_vals = self.C2(states).gather(1, actions.view(-1,1)).view(1,-1)[0]
        q1_loss = torch.nn.functional.mse_loss(q1_vals, y)
        q2_loss = torch.nn.functional.mse_loss(q2_vals, y)
        self.C1_optim.zero_grad()
        self.C2_optim.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.C1_optim.step()
        self.C2_optim.step()

        # Update policy if this is a policy training step
        if self.steps % self.update_every == 0:
            # Loss for the policy network
            act_probs, log_probs = self.policy(states)
            with torch.no_grad():
                q1 = self.C1(states)
                q2 = self.C2(states)
                min_q_vals = torch.min(q1, q2)

            # Policy loss using the full expectation
            pol_loss = (act_probs * (self.alpha * log_probs - min_q_vals)).sum(dim=1).mean()
            self.pol_optim.zero_grad()
            pol_loss.backward()
            self.pol_optim.step()

        # Update target networks using soft update
        self.update_target()


    def train(self):
        self.steps = 0
        while self.steps < self.max_steps:
            done, truncated = False, False
            obs, _ = self.env.reset()
            while not (done or truncated): # Until training env stops
                # Evaluate 
                if (self.steps % self.pt == 0) and self.evaluate:
                    self.eval()
                    print(self.reward_log[-1])
                
                # Sample action from the policy
                with torch.no_grad():
                    action = self.sample_action(observation=obs)
                next_obs, reward, done, truncated, _ = self.env.step(action)
                self.mem.save(obs, action, next_obs, reward, (done or truncated)) # Save to replay buffer
                obs = next_obs
                self.steps += 1

                # Start training after initial sample is completed
                if self.steps > self.init_sample:
                    self.train_batch()
                
            self.env.close()


    def save(self, filename):
        # Save policy parameters.
        torch.save(self.policy.state_dict(), filename)


    def load(self, filename):
        # Load policy parameters
        self.policy.load_state_dict(torch.load(filename))