import numpy as np
import torch
from torch import nn
import random

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)

class NoneActFun(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

def relu_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(0, 1)

def sigmoid_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def ActInit(net, act):
    if act == 'Sine':
        net.apply(sine_init)
        net[0].apply(first_layer_sine_init)
    elif act == 'ReLU':
        net.apply(relu_init)
    elif act == 'LeakyReLU':
        net.apply(relu_init)
    elif act == 'Sigmoid':
        net.apply(sigmoid_init)
    elif act == 'Tanh':
        net.apply(sigmoid_init)
    else:
        net.apply(relu_init)

def Activation(act):
    if act == 'Sine':
        act_fun = Sine()
    elif act == 'ReLU':
        act_fun = nn.ReLU()
    elif 'LeakyReLU' in act:
        negative_slope = float(act[9:])
        act_fun = nn.LeakyReLU(negative_slope)
    elif act == 'Sigmoid':
        act_fun = nn.Sigmoid()
    elif act == 'Tanh':
        act_fun = nn.Tanh()
    elif act == 'None':
        act_fun = NoneActFun()
    else:
        raise NotImplemented
    return act_fun

class MLP(nn.Module):
    def __init__(self, input=1, hidden=256, output=1, layer=5, act='Sine', output_act=False, **kwargs):
        super().__init__()
        self.net=[]
        self.hyper = {'input':input, 'output':output, 'hidden':hidden, 'layer':layer, 'act':act, 'output_act':output_act}
        # Activation
        act_fun = Activation(act)
        # Network structure
        self.net.append(nn.Linear(input, hidden))
        self.net.append(act_fun)
        for i in range(layer-2):
            self.net.append(nn.Linear(hidden, hidden))
            self.net.append(act_fun)
        # Output activation
        if layer >= 2:
            self.net.append(nn.Linear(hidden, output))
            if output_act == True:
                self.net.append(act_fun)
        self.net=nn.Sequential(*self.net)
        # Parameter initialization
        ActInit(self.net, act)
        print(self.net)
    def forward(self, input):
        output = self.net(input)
        return output

class CONV(nn.Module):
    def __init__(self, input=1, in_channel=1, out_channel=1, height=28, width=28, pool='Ave', act='Sine', **kwargs):
        super().__init__()
        self.net=[]
        act_fun = Activation(act)
        y = torch.ones(1,input)
        # Network structure
        self.net.append(nn.Linear(input, in_channel*height*width))
        y = self.net[-1](y)
        self.net.append(act_fun)
        y = self.net[-1](y)

        self.net.append(nn.Unflatten(-1, (in_channel, height, width)))
        y = self.net[-1](y)
        self.net.append(nn.Conv2d(in_channel, out_channel, kernel_size=(2,2), stride=(1,1), bias=True))
        y = self.net[-1](y)
        self.net.append(torch.nn.Dropout2d())
        y = self.net[-1](y)
        self.net.append(act_fun)
        y = self.net[-1](y)
        
        if pool == 'Max':
            self.net.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))  # return_indices=False
        elif pool == 'Ave':
            self.net.append(nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)))
        y = self.net[-1](y)
        self.net.append(act_fun)
        y = self.net[-1](y)

        self.net.append(nn.Flatten())
        y = self.net[-1](y)
        self.net.append(torch.nn.Dropout())
        y = self.net[-1](y)
        self.net.append(nn.Linear(y.shape[-1], 1))
        y = self.net[-1](y)

        self.net=nn.Sequential(*self.net)
        print(self.net)
    def forward(self, input):
        output = self.net(input)
        return output
    
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            #The size of the picture is 28x28
            torch.nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 3,stride = 1,padding = 1),
            nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size = 2,stride = 2),
            
            #The size of the picture is 14x14
            torch.nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size = 3,stride = 1,padding = 1),
            # torch.nn.Dropout2d(),
            nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size = 2,stride = 2),
            
            #The size of the picture is 7x7
            torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            nn.Tanh(),
            
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 7*7*64,out_features = 128),
            nn.Tanh(),
            # torch.nn.Dropout(),
            torch.nn.Linear(in_features = 128,out_features = 1)
            # torch.nn.Softmax(dim=1)
        )
        print(self.net)
        
    def forward(self,input):
        output = self.net(input)
        return output

class Network(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.net = torch.nn.Sequential()
        for module in modules:
            self.net.append(module=ModuleDict(module))
        self.net.apply(sine_init)
        print(self.net)

    def forward(self,input):
        output = self.net(input)
        return output

def ModuleDict(module):
    if 'Linear' in module:
        in_features, out_features = [int(var) for var in module.split('_')[1:]]
        return torch.nn.Linear(in_features=int(in_features), out_features=int(out_features))
    elif 'Conv2d' in module:
        in_channels, out_channels, kernel_size, stride, padding = [int(var) for var in module.split('_')[1:]]
        return torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding) 
    elif 'AvgPool2d' in module:
        kernel_size, stride = [int(var) for var in module.split('_')[1:]]
        return torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    elif 'MaxPool2d' in module:
        kernel_size, stride = [int(var) for var in module.split('_')[1:]]
        return torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    elif 'Dropout2d' in module:
        return torch.nn.Dropout2d()
    elif 'Dropout' in module:
        return torch.nn.Dropout()
    elif 'Flatten' in module:
        return torch.nn.Flatten()
    elif 'Unflatten' in module:
        return torch.nn.Unflatten()
    else:
        return Activation(module)