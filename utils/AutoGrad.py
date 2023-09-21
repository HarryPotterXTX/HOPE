import math
import torch

def cald(y,x,idx):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0][:, idx].reshape(-1, 1)

def cal_grad(y:torch.tensor, x:torch.tensor, idx:tuple=(0), order:int=10):
    '''calculate the n-order unmixed partial derivatives'''
    grad_list = [y]
    for k in range(1, order+1):
        grad = torch.autograd.grad(grad_list[-1], x, grad_outputs=torch.ones_like(grad_list[-1]), create_graph=True)[0][idx]
        grad_list.append(grad)
        # print(grad)
    return grad_list

def taylor_output(grads:list, dx:float, order:int=10):
    y = grads[0]
    for k in range(1,order+1):
        y = y + grads[k]/math.factorial(k)*(dx**k)
    return y