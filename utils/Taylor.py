import copy
from matplotlib.pyplot import delaxes
import torch
import numpy as np
from typing import Dict, List
import random
import time
import math
import torch.nn.functional as F
from utils.ChainMatrix import ChainTransMatrix, ChainCal, FormulaCal
from utils.Activation import JudgeAct

def Gsum(a, q, n):  # Sum of geometric series
    return a*n if q == 1 else int(a*(q**n-1)/(q-1))

def CalMatAct(Module, MatCal:Dict, input:torch.tensor, block:bool=False):
    """y=act(y): nonlinear activation function, such as Sine(), Tanh(), Sigmoid()...; block=True means we store M with a dict
    """
    order = len(MatCal)
    act = JudgeAct(Module, order)
    if not block and len(input.shape)>2:    # fc: batch num; conv: batch channel height width
        input = input.view(input.shape[0], -1)
    if 'Sigmoid' in str(type(Module)):      # Act.diff(input, i), the input parameter is the output of the activation function for Sigmoid and Tanh
        input = torch.sigmoid(input)
    elif 'Tanh' in str(type(Module)):   
        input = torch.tanh(input)
    Beta = {}
    for i in range(1, order+1):
        Beta[i] = act.diff(input, i)        # Beta_i = diag(act.diff) it is not urgent to convert
    if block:                               # store M with a dict
        M = {}
        for key in MatCal.keys():
            M[key] = {}
            for sub_key in MatCal[key].keys():
                formula = MatCal[key][sub_key]
                data = FormulaCal(formula, Beta)    # act.diff is used rather than diag(act.diff) for faster calculation
                M[key][sub_key] = data              # Matrix multiplication for diag(data), or dot multiplication for data
    else:
        batch, row = Beta[1].shape          # batch row
        M = torch.zeros((batch, row*order, row*order))
        idx = torch.LongTensor([i for i in range(row)])
        for key in MatCal.keys():
            for sub_key in MatCal[key].keys():
                formula = MatCal[key][sub_key]
                data = FormulaCal(formula, Beta)    # act.diff is used rather than diag(act.diff) for faster calculation
                M[:,row*(int(key)-1)+idx,row*(int(sub_key)-1)+idx] = data   # idxs-based assignment is much faster than block assignment
    return M

def CalMatLin(Module, MatCal:Dict, input:torch.tensor, mixed:False, block:bool=False):
    """y=Wx+b: linear layer
    """
    order = len(MatCal)
    W = Module.weight
    dy, dx = W.shape
    batch = input.shape[0]
    if not mixed:
        M = {} if block else torch.zeros((batch, int(order*dx), int(order*dy)))         # not include mixed partial derivatives
        for k in range(1, order+1):
            if block:
                M[k] = torch.pow(W.T, k)
            else:
                x1, x2, y1, y2 = int(dx*(k-1)), int(dx*k), int(dy*(k-1)), int(dy*k)
                M[:,x1:x2,y1:y2] = torch.pow(W.T, k)
    else:
        M = {} if block else torch.zeros((batch, Gsum(dx, dx, order), int(order*dy)))   # dx+dx^2+...+dx^n=(dx^(n+1)-dx)/(dx-1)
        Wn = torch.ones(1, dy).to(W.device)
        for k in range(1, order+1):
            augW1 = torch.kron(W.T.contiguous(), torch.ones(dx**(k-1), 1).to(W.device))  
            augW2 = torch.kron(torch.ones(dx, 1).to(W.device), Wn) 
            Wn = torch.mul(augW1, augW2)
            if block:
                M[k] = Wn
            else:
                x1, x2, y1, y2 = Gsum(dx, dx, k-1), Gsum(dx, dx, k), int(dy*(k-1)), int(dy*k)
                M[:,x1:x2,y1:y2] = Wn
    return M

class TaylorNetWork():
    def __init__(self, order:int, network):
        self.order = order
        self.MatCal = ChainTransMatrix(order)   # the n-order chain transformation matrix expression
        self.network = network.eval()
        self.net = network.net
        self.layer = len(self.net)

    def forward(self, input, mixed:bool=False):
        time_start = time.time()
        self.workpoint = input
        self.module_info = []
        for i in range(len(self.net)):
            Module = self.net[i]
            info = {'name':str(type(Module))}
            output = Module.forward(input)
            # cal the n-order chain transformation matrix
            if len(input.shape) >= 4:   # conv layer and its activation layer: batch channel height width
                if 'Conv2d' in str(type(Module)):                                                 
                    output_padding = ((input.shape[-2]-Module.kernel_size[0])%Module.stride[0], (input.shape[-1]-Module.kernel_size[1])%Module.stride[1])
                    info.update({'W':Module.weight, 'padding':Module.padding, 'stride':Module.stride, 'dilation':Module.dilation, \
                                    'output_padding': output_padding})
                elif 'MaxPool2d' in str(type(Module)):   
                    Module.return_indices=True
                    output, indices = Module.forward(input)
                    info.update({'indices':indices, 'kernel_size':Module.kernel_size, 'stride':Module.stride})
                elif 'AvgPool2d' in str(type(Module)):
                    Module.kernel_size = (Module.kernel_size, Module.kernel_size) if isinstance(Module.kernel_size, int) else Module.kernel_size
                    Module.stride = (Module.stride, Module.stride) if isinstance(Module.stride, int) else Module.stride
                    output_padding = ((input.shape[-2]-Module.kernel_size[0])%Module.stride[0], (input.shape[-1]-Module.kernel_size[1])%Module.stride[1])
                    weight = torch.zeros(input.shape[-3],input.shape[-3],Module.kernel_size[0],Module.kernel_size[1])
                    for i in range(weight.shape[0]):
                        weight[i][i] = torch.ones(Module.kernel_size[0],Module.kernel_size[1])/(Module.kernel_size[0]*Module.kernel_size[1])
                    info.update({'W':weight, 'padding':Module.padding, 'stride':Module.stride, 'output_padding': output_padding})
                elif 'Flatten' in str(type(Module)) or 'Dropout2d' in str(type(Module)):
                    pass
                else:
                    info['M'] = CalMatAct(Module=Module, MatCal=self.MatCal, input=input, block=True)  # matrix is too large that we express it with a dict
            else:                       # fc layer and its activation layer: batch row col(1)
                if 'linear' in str(type(Module)):
                    mixed = True if (i==0 and mixed==True) else False    
                    info['M'] = CalMatLin(Module=Module, MatCal=self.MatCal, input=input, mixed=mixed, block=True)
                elif 'Unflatten' in str(type(Module)) or 'Dropout' in str(type(Module)):
                    pass
                else:
                    info['M'] = CalMatAct(Module=Module, MatCal=self.MatCal, input=input, block=True) 
            info.update({'in_shape':input.shape, 'out_shape':output.shape, 'in_size':torch.numel(input[0]), 'out_size':torch.numel(output[0])})
            self.module_info.append(info)
            input = output
        self.output = input
        # print(f'time cost of forward propgation: {time.time()-time_start}')
        return self.module_info

    def backward(self, device:str='cpu'):
        time_start = time.time()
        v = torch.zeros((self.workpoint.shape[0],self.order,1)).to(device)  # the first n order derivatives, batch order 1
        v[:,0,0] = 1
        self.derivatives = [v]
        for i in range(len(self.module_info)-1,-1,-1):
            info = self.module_info[i]
            v = self.derivatives[-1]                        # fc: batch order*num; conv: batch order*channel*height*width
            if len(info['in_shape']) >= 4:                  # conv layer and its activation layer: batch channel height width
                if 'Conv2d' in info['name']:
                    v = CalDriConv(info, v, self.order)
                elif 'MaxPool2d' in info['name']:   
                    v = CalDriMaxPool(info, v, self.order)
                elif 'AvgPool2d' in info['name']:   
                    v = CalDriConv(info, v, self.order)     # Average pooling is similar to convolution
                elif 'Flatten' in info['name'] or 'Dropout2d' in info['name']:
                    pass
                else:
                    v = CalDriAct(info, v, self.order)
            else:                                           # linear layer or nonlinear activation layer
                if 'linear' in info['name']:
                    v = CalDriLin(info, v, self.order)
                elif 'Unflatten' in info['name'] or 'Dropout' in info['name']:
                    pass
                else:
                    v = CalDriAct(info, v, self.order)
            self.derivatives.append(v)
        alpha = [self.output.to(device)]
        for i in range(v.shape[1]):                         # An[0]: batch row col(1)
            alpha.append(v[:,i].to(device))
        # print(f'time cost of backpropgation: {time.time()-time_start}')
        return alpha

def CalDriMaxPool(info, vo_flatten, order):
    indices, kernel_size, stride, in_shape, out_shape, in_size = info['indices'], info['kernel_size'], info['stride'], \
        info['in_shape'], info['out_shape'], info['in_size']
    batch = vo_flatten.shape[0]
    vx_flatten = torch.zeros(batch, in_size*order, 1).to(vo_flatten.device)
    vo = vo_flatten.view(batch, order, out_shape[-3], out_shape[-2], out_shape[-1])   # batch (order channel height width)
    for k in range(1, order+1):
        vo_k = vo[:,k-1]
        vx_k = F.max_unpool2d(input=vo_k, indices=indices , kernel_size=kernel_size, stride=stride, output_size=in_shape)
        vx_flatten[:,int((k-1)*in_size):int(k*in_size)] = vx_k.view(batch, in_size, 1)
    return vx_flatten
    
def CalDriLin(info, vo_flatten, order):
    M, in_size, out_size = info['M'], info['in_size'], info['out_size']
    # M is a diagonal matrix. Using a dict to store the required matrix blocks can greatly save memory and speed up calculation
    if isinstance(M, dict): 
        batch = vo_flatten.shape[0]
        vo = vo_flatten.view(batch, order, out_size, 1)     # batch (order num)
        if M[1].shape == M[order].shape:                    # not include mixed partial derivatives
            vx_flatten = torch.zeros(batch, in_size*order, 1).to(vo_flatten.device)
            for k in range(1, order+1):
                x1, x2 = int((k-1)*in_size), int(k*in_size)
                vx_flatten[:,x1:x2] = torch.matmul(M[k], vo[:,k-1]).view(batch, in_size, 1)
        else:
            vx_flatten = torch.zeros(batch, Gsum(in_size, in_size, order), 1).to(vo_flatten.device)
            for k in range(1, order+1):
                x1, x2 = Gsum(in_size, in_size, k-1), Gsum(in_size, in_size, k)
                vx_flatten[:,x1:x2] = torch.matmul(M[k], vo[:,k-1]).view(batch, in_size**k, 1)
    else:
        vx_flatten = torch.matmul(info['M'], vo_flatten)    # v[m] = M[m+1]v[m+1]
    return vx_flatten
    
def CalDriAct(info, vo_flatten, order):
    M, out_shape, in_size, out_size = info['M'], info['out_shape'], info['in_size'], info['out_size']
    # M is a lower triangular matrix. Using a dict to store the required matrix blocks can greatly save memory and speed up calculation
    if isinstance(M, dict): 
        batch = vo_flatten.shape[0]
        vx_flatten = torch.zeros(batch, in_size*order, 1).to(vo_flatten.device)
        if len(info['in_shape']) >= 4:  # nonlinear activation function for conv layer
            vo = vo_flatten.view(batch, order, out_shape[-3], out_shape[-2], out_shape[-1])     # batch (order channel height width)
        else:
            vo = vo_flatten.view(batch, order, out_size)                                        # batch (order num)
        for k in range(1, order+1):
            for q in range(1, k+1):
                vx_flatten[:,int((k-1)*in_size):int(k*in_size)] += torch.mul(M[k][q], vo[:,q-1]).view(batch, in_size, 1)
    else:
        vx_flatten = torch.matmul(info['M'], vo_flatten)                                        # v[m] = M[m+1]v[m+1]
    return vx_flatten

def CalDriConv(info, vo_flatten, order):
    W, padding, stride, output_padding, out_shape, in_size = info['W'], info['padding'], info['stride'], \
        info['output_padding'], info['out_shape'], info['in_size']
    batch = vo_flatten.shape[0]
    vx_flatten = torch.zeros(batch, in_size*order, 1).to(vo_flatten.device)
    vo = vo_flatten.view(batch, order, out_shape[-3], out_shape[-2], out_shape[-1])   # batch (order channel height width)
    for k in range(1, order+1):
        vo_k = vo[:,k-1]
        weight = torch.pow(W, k)
        # o = conv(x, f): a^ky/ax^k = conv(a^ky/ao^k, rotate(W^(ok)))
        vx_k = F.conv_transpose2d(input=vo_k, weight=weight, padding=padding, stride=stride, output_padding=output_padding)
        vx_flatten[:,int((k-1)*in_size):int(k*in_size)] = vx_k.view(batch, in_size, 1)
    return vx_flatten

def list2dict(alpha_list:list, n_in:int, alpha_dict:Dict={}, idx:int=0, key:str='', mixed:bool=False): 
    if idx >= len(alpha_list):
        return alpha_dict
    alpha_dict[key] = alpha_list[idx]
    for var in range(1,n_in+1):
        if key == '':
            new_idx = var
            new_key = key + str(var)
        elif not mixed:
            if key.split(',')[-1] == str(var): # less than 10 input
                new_idx = idx + n_in 
                new_key = key + ',' + str(var)
            else:
                continue
        else:
            new_idx = idx*n_in + var   # n_in-base(no zero)  eg. 1213(n_in=3)-> idx = 1*3^3+2*3^2+1*3^1+3*3 
            new_key = key + ',' + str(var)
        list2dict(alpha_list, n_in, alpha_dict, new_idx, new_key, mixed)
    return alpha_dict

def taylor_expansion_dict(alpha:Dict, input:np.matrix, work_point:np.array, mixed:bool=False):
    """
    input: (batch, index)
    """
    output = 0
    for key in alpha.keys():
        if key == '':
            y = np.ones((input.shape[0],1))*alpha[key]
        else:
            vars = [int(var) for var in key.split(',')]
            y = np.ones((input.shape[0],1))*alpha[key]/math.factorial(len(vars))
            if not mixed:
                if vars.count(vars[0]) == len(vars):
                    y = np.multiply(y, np.power((input[:,vars[0]-1]-work_point[vars[0]-1]),len(vars)))
                else:
                    continue
            else:
                flag = []
                for var in vars:
                    if var not in flag:
                        power = vars.count(var)
                        y = np.multiply(y, np.power((input[:,var-1]-work_point[var-1]), power))
                        flag.append(var)
        output = output + y
    return output

def taylor_expansion_list(alpha_list:list, input:np.matrix, work_point:np.array, idx:int=0, indexs:list=[], order:int=0): 
    """
    input: (batch, index)
    """
    if idx >= len(alpha_list):
        return 0
    output = np.ones((input.shape[0],1))*alpha_list[idx]/math.factorial(order)
    for index in indexs:
        output = np.multiply(output,(input[:,index-1]-work_point[index-1]))
    n_in = input.shape[1]
    for var in range(1,n_in+1):
        new_idx = idx*n_in + var   # n_in-base(no zero)  eg. 1213(n_in=3)-> idx = 1*3^3+2*3^2+1*3^1+3*3 
        new_indexs = indexs + [var]
        new_order = order+1
        output += taylor_expansion_list(alpha_list, input, work_point, new_idx, new_indexs, new_order)
    return output