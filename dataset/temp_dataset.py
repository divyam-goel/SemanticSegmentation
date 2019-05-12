from random import choice
from string import ascii_lowercase

import torch
from torch.utils.data.dataset import Dataset


class MyCustomDataset(Dataset):
    def __init__(self, input_size=(3, 321, 321), count=50):
    	self.size = input_size
    	self.count = count
    	
    	self.images = []
    	self.labels = []
    	self.names = []
    	for i in range(self.count):
    		self.images.append(torch.empty(self.size).uniform_(0, 1))
    		self.labels.append(torch.randint(3, 5, self.size[1:]))
    		self.names.append(''.join(choice(ascii_lowercase) for i in range(8)))
        
    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.size, self.names[index]

    def __len__(self):
        return self.count 