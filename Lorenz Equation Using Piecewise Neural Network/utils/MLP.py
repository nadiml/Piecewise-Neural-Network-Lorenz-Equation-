import torch 
import torch.nn as nn 

import numpy as np 
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int, 
                 output_dim: int, 
                 num_hidden_layers: int, 
                 num_neurons: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = num_hidden_layers
        self.n_neurons = num_neurons 

        layers = []
        layers.append(nn.Linear(input_dim, self.n_neurons))
        layers.append(nn.Tanh())

        for _ in range(self.n_hidden):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(self.n_neurons, output_dim))
        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x):
        return self.model(x)
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


