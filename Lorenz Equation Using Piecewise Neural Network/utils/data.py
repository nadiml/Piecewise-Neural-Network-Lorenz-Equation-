import torch
from torch.utils.data import Dataset, DataLoader
import math
import random

class MyDataset(Dataset):
    def __init__(self, start:float, end:float, total_points:int, num_subdomain:int = 1):
        super().__init__()

        self.start = start
        self.end = end 
        self.total_points = total_points
        self.num_subdomain = num_subdomain
        
        assert total_points % num_subdomain == 0 ,'Number of points must be divisible by number of subdomain'

        self.points_per_subdomain = total_points // num_subdomain

        self.data = {}

        for i in range(num_subdomain):
            subdomain_start = start + (end-start)*i /num_subdomain
            subdomain_end = start + (end - start)*(i+1)/num_subdomain

            x = self.sub_domain_generator(subdomain_start, subdomain_end, self.points_per_subdomain)
            self.data[i] = x

    def __len__(self):
        return self.num_subdomain
    

    def __getitem__(self, key):
        if key not in self.data:
            raise KeyError(f'Subdomain {key} does not exist')
        
        subdomain_data = self.data[key]
        indices = torch.randperm(subdomain_data.size(0))
        shuffled_data = subdomain_data[indices]
        return shuffled_data
        


    def sub_domain_generator(self, a, b, num_points = 200, a_inc = True, b_inc = True):
        sub_data = torch.linspace(a,b,num_points).reshape(-1, 1)
        return sub_data


class SubdomainDataLoader(DataLoader):
    def __init__(self, dataset, num_works = 0):
        super().__init__(dataset, batch_size=1, num_workers=num_works, collate_fn=self.collate_fn)


    def collate_fn(self, batch):
        return batch[0]
    

class DataSampler:
    def __init__(self, 
                 batch_size: int, 
                 epochs: int,
                 data: torch.tensor, 
                 device: str='cpu'):
        
        self.coords = data.to(device)
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
