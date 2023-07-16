import math
import torch
from torch import pi as pi
from torch import sin, cos, exp
import numpy as np

def create_flattened_coords(shape) -> torch.Tensor:
    parameter = []
    dim = 1
    for i in range(len(shape)):
        minimum,maximum,num = shape[i]
        parameter.append(torch.linspace(minimum,maximum,num))
        dim *= num
    coords = torch.stack(torch.meshgrid(parameter),axis=-1)
    flattened_coords = coords.reshape(dim,len(shape))
    return flattened_coords

def normalize_data(data, scale_min, scale_max, data_min:float=None, data_max:float=None):
    if data_min==None or data_max==None:
        data_min, data_max = data.min(), data.max()
    if scale_min=='none' or scale_max=='none':
        side_info = {'scale_min':None, 'scale_max':None, 'data_min':float(data_min), 'data_max':float(data_max)}
    else:
        data = (data - data_min)/(data_max - data_min)
        data = data*(scale_max - scale_min) + scale_min
        side_info = {'scale_min':float(scale_min), 'scale_max':float(scale_max), 'data_min':float(data_min), 'data_max':float(data_max)}
    return data, side_info

def invnormalize_data(data, scale_min, scale_max, data_min, data_max):
    if scale_min!=None and scale_max!=None:
        data = (data - scale_min)/(scale_max - scale_min)
        data = data*(data_max - data_min) + data_min
    return data

class BaseSampler:
    def __init__(self, batch_size: int, epochs:int, device:str='cpu'):
        self.batch_size = int(batch_size)
        self.epochs = epochs
        self.device = device
        self.evaled_epochs = []

    def judge_eval(self, eval_epoch):
        if self.epochs_count%eval_epoch==0 and self.epochs_count!=0 and not (self.epochs_count in self.evaled_epochs):
            self.evaled_epochs.append(self.epochs_count)
            return True
        elif self.index>=self.pop_size and self.epochs_count>=self.epochs-1:
            self.epochs_count = self.epochs
            return True
        else:
            return False

    def __len__(self):
        return self.epochs*math.ceil(self.pop_size/self.batch_size)

    def __iter__(self):
        self.index = 0
        self.epochs_count = 0
        return self

# 1. y=e^(kx), x in [-1,1]
class ExpSampler(BaseSampler):
    def __init__(self, batch_size: int, epochs:int, device:str='cpu', normal_min:float=-1, normal_max:float=1, k:float=1, r1:float=-1, r2:float=1) -> None:
        super().__init__(batch_size=batch_size, epochs=epochs, device=device)
        self.shape = [[r1,r2,2048*8]]
        self.coords = create_flattened_coords(self.shape).to(device)
        self.label = torch.exp(k*self.coords)
        self.label, self.side_info = normalize_data(self.label, scale_min=normal_min, scale_max=normal_max)
        self.label = self.label.to(device)
        self.pop_size = self.label.shape[0]
        
    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            # sampled_idxs = torch.cat([sampled_idxs,torch.tensor([-1,0,1])],dim=-1) # boundary
            sampled_coords = self.coords[sampled_idxs, :]
            sampled_label = self.label[sampled_idxs, :]
            self.index += self.batch_size
            return sampled_coords, sampled_label
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration

# 2. y=x**2, x in [-1,1]
class PolySampler(BaseSampler):
    def __init__(self, batch_size: int, epochs:int, device:str='cpu', normal_min:float=-1, normal_max:float=1) -> None:
        super().__init__(batch_size=batch_size, epochs=epochs, device=device)
        self.shape = [[-1,1,2048]]
        self.coords = create_flattened_coords(self.shape).to(device)
        self.label = self.coords**2
        self.label, self.side_info = normalize_data(self.label, scale_min=normal_min, scale_max=normal_max)
        self.label = self.label.to(device)
        self.pop_size = self.label.shape[0]
        
    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            # sampled_idxs = torch.cat([sampled_idxs,torch.tensor([-1,0,1])],dim=-1) # boundary
            sampled_coords = self.coords[sampled_idxs, :]
            sampled_label = self.label[sampled_idxs, :]
            self.index += self.batch_size
            return sampled_coords, sampled_label
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration

# 3. y=sin(x), x in [-pi,pi]
class SinSampler(BaseSampler):
    def __init__(self, batch_size: int, epochs:int, device:str='cpu', normal_min:float=-1, normal_max:float=1) -> None:
        super().__init__(batch_size=batch_size, epochs=epochs, device=device)
        self.shape = [[-4*pi,4*pi,4*2048]]
        self.coords = create_flattened_coords(self.shape).to(device)
        self.label = torch.sin(self.coords)
        self.label, self.side_info = normalize_data(self.label, scale_min=normal_min, scale_max=normal_max)
        self.label = self.label.to(device)
        self.pop_size = self.label.shape[0]
        
    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            # sampled_idxs = torch.cat([sampled_idxs,torch.tensor([-1,0,1])],dim=-1) # boundary
            sampled_coords = self.coords[sampled_idxs, :]
            sampled_label = self.label[sampled_idxs, :]
            self.index += self.batch_size
            return sampled_coords, sampled_label
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration

# 4. y=x**3, x in [-1,1]
class Poly1Sampler(BaseSampler):
    def __init__(self, batch_size: int, epochs:int, device:str='cpu', normal_min:float=-1, normal_max:float=1) -> None:
        super().__init__(batch_size=batch_size, epochs=epochs, device=device)
        self.shape = [[-1,1,2048]]
        self.coords = create_flattened_coords(self.shape).to(device)
        self.label = self.coords**3
        self.label, self.side_info = normalize_data(self.label, scale_min=normal_min, scale_max=normal_max)
        self.label = self.label.to(device)
        self.pop_size = self.label.shape[0]
        
    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            sampled_coords = self.coords[sampled_idxs, :]
            sampled_label = self.label[sampled_idxs, :]
            self.index += self.batch_size
            return sampled_coords, sampled_label
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration
        
# 5. y=f(x1, x2)=(x1^2+x2)/2, x in [-1,1]
class DiscoverySampler(BaseSampler):
    def __init__(self, batch_size: int, epochs:int, device:str='cpu', normal_min:float=-1, normal_max:float=1) -> None:
        super().__init__(batch_size=batch_size, epochs=epochs, device=device)
        r1, r2 = -1, 1
        self.shape = [[r1,r2,512], [r1,r2,512]]
        self.coords = create_flattened_coords(self.shape).to(device)
        self.label = (self.coords[:,0:1]**2 + self.coords[:,1:2])/2
        self.label, self.side_info = normalize_data(self.label, scale_min=normal_min, scale_max=normal_max)
        self.label = self.label.to(device)
        self.pop_size = self.label.shape[0]
        
    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            sampled_coords = self.coords[sampled_idxs, :]
            sampled_label = self.label[sampled_idxs, :]
            self.index += self.batch_size
            return sampled_coords, sampled_label
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration
        
SamplerDict = {'Exp':ExpSampler, 'Poly': PolySampler, 'Sin':SinSampler, 
               'Poly1': Poly1Sampler, 'Discovery':DiscoverySampler}