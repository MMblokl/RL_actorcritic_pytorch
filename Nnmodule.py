import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    # Polci network
    def __init__(self, n_obs: int, n_act: int, n_neurons: int, n_layers: int, device):
        super(Policy, self).__init__()
        # Input layer
        self.input = nn.Linear(n_obs, n_neurons)
        
        # Variable length sequential hidden layer.
        self.hidden = []
        for i in range(n_layers):
            self.hidden.append(nn.Linear(n_neurons, n_neurons))
            self.hidden.append(nn.ReLU()) # ReLU activation between each layer
        self.hidden = nn.Sequential(*self.hidden)
        
        # Output layer that outputs the action to take given the state
        self.output = nn.Linear(n_neurons, n_act)

        # Set the entire policy net onto the currently used device
        self.to(device)
    
    def forward(self, x):
        # Forward pass through network
        x = self.input(x) # Input layer
        x = self.hidden(x)
        x = self.output(x) # Ouput layer
        return F.softmax(x) # Softmax output to make the probabilities to 1


class Value(nn.Module):
    # Critic network
    def __init__(self, n_obs: int, n_act: int, n_neurons: int, n_layers: int, device):
        super(Value, self).__init__()
        # Layer that takes the currect state, or current observation as input
        self.input = nn.Linear(n_obs, n_neurons)
        
        # Sequential hidden layer
        self.hidden = []
        for i in range(n_layers):
            self.hidden.append(nn.Linear(n_neurons, n_neurons))
            self.hidden.append(nn.ReLU())
        self.hidden = nn.Sequential(*self.hidden)
        
        # Output layer with a single output, which is the value estimate for the given state.
        self.output = nn.Linear(n_neurons, n_act)

        # Set the GPU to use for training if possible, otherwise just use the CPU
        self.to(device)
    
    def forward(self, x):
        # Forward pass through network
        x = self.input(x) # Input layer
        x = self.hidden(x)
        val = self.output(x) # Ouput layer
        return val

