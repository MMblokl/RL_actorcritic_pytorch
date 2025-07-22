import torch
import numpy as np
from AC import AC

class A2C(AC):
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
        # Inherit all initiated values and class objects from AC.
        super().__init__(
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            n_neurons=n_neurons,
            n_layers=n_layers,
            n_step=n_step,
            envname=envname,
            device=device,
            max_steps=max_steps,
            plot_rate=plot_rate,
            n_reps=n_reps,
            evaluate=evaluate
            )

    # Replace the training function from AC to use advantage.
    def train_ep(self, log_probs, rewards, critic_vals):
        # Set the returns and log probs as a single tensor for easy multiplication.
        returns = self.calculate_returns(rewards=rewards)
        log_probs = torch.stack(log_probs)
        critic_vals = torch.stack(critic_vals)
        q_values = self.calc_q(returns=returns, critic_vals=critic_vals) # Q return value
        q_values = (q_values - q_values.mean()) / (q_values.std() + np.finfo(np.float32).eps.item()) # normalize q for unstable learning avoidance

        # Advantage: Q(st, at) - V(st)
        advantage = q_values - critic_vals

        # Use the log probabily and the return value to calculate the loss.
        # Detach the tensor to not let the critic and policy network gradients mix
        policy_loss = -(log_probs * advantage.detach()).sum()
        critic_loss = torch.nn.functional.mse_loss(critic_vals, q_values)
        self.pol_optim.zero_grad()
        self.cri_optim.zero_grad()
        policy_loss.backward()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.pol_optim.step()
        self.cri_optim.step()
