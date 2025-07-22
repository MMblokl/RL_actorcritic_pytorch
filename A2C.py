import torch
import numpy as np
from AC import AC

class A2C(AC):
    def __init__(self, gamma, alpha, beta, n_neurons, n_layers, n_step, env, device):
        # Inherit all initiated values and class objects from AC.
        super().__init__(
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            n_neurons=n_neurons,
            n_layers=n_layers,
            n_step=n_step,
            env=env,
            device=device
            )

    # Replace the training function from AC to use advantage.
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
