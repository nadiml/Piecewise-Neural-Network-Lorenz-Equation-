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

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dim, self.n_neurons))
        self.layers.append(nn.Tanh())

        for _ in range(self.n_hidden - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.n_neurons, self.n_neurons), 
                    nn.LayerNorm(self.n_neurons),
                    nn.Tanh(), 
                    nn.Dropout(0.1)

                )
            )

        self.final = nn.Linear(self.n_neurons, self.output_dim)

        self._initialize_weights()

    def forward(self, t):
        x = t
        residual = None
        
        for i, layer in enumerate(self.layers):
            if i > 0 and i % 2 == 0 and residual is not None:
                x = x + residual  # Residual connection
            x = layer(x)
            residual = x
            
        x = self.final(x)

        return x 
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


