import torch
import math


def create_flattened_coords(shape) -> torch.Tensor:
    parameter = []
    dim = 1
    for i in range(len(shape)):
        minimum,maximum,num = shape[i]
        parameter.append(torch.linspace(minimum,maximum,num))
        dim *= num
    coords = torch.stack(torch.meshgrid(parameter, indexing='ij'),axis=-1)
    flattened_coords = coords.reshape(dim,len(shape))
    return flattened_coords

class DataSampler:
    def __init__(self, 
                 batch_size: int, 
                 epochs: int,
                 shape: list, 
                 device: str='cpu'):
        
        self.shape = shape
        self.coords = create_flattened_coords(self.shape).to(device)
        self.batch_size = int(batch_size)
        self.pop_size = self.coords.shape[0]
        self.epochs = epochs
        self.devie = device


    def __len__(self):
        return self.epochs*math.ceil(self.pop_size/self.batch_size)
    
    def __iter__(self):
        self.index = 0 
        self.epochs_count = 0
        return self
    
    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            sampled_coords = self.coords[sampled_idxs, :]            
            self.index += self.batch_size
            return sampled_coords
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration