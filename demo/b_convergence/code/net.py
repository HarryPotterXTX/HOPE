import torch
import random
import math
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from utils.Network import MLP
from utils.Logger import reproduc

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # num_input = math.sqrt(m.weight.shape[0]*m.weight.shape[1])
            m.weight.uniform_(-1/num_input*w0, 1/num_input*w0)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # num_input = math.sqrt(m.weight.shape[0]*m.weight.shape[1])
            m.weight.uniform_(-1/num_input*w0, 1/num_input*w0)

if __name__=='__main__':
    reproduc()
    c, d, f, l = 1, 1, 512, 5
    act = 'Sine'
    output_act = True
    global w0
    w0_list = [0.010,0.100,1.000,10.00,100.0]
    for w0 in w0_list:
        net = MLP(input=c, hidden=f, output=d, layer=l, act=act, output_act=output_act)
        net.net.apply(sine_init)
        net.net[0].apply(first_layer_sine_init)
        save_path = os.path.join('demo/b_convergence/outputs',act+f'{w0}_{c}_{d}_{f}_{l}_True','net.pt')
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        torch.save(net, save_path)