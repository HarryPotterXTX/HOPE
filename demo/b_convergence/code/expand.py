import torch
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from utils.Taylor import TaylorNetWork

def main(reference_point, order, net_path, mixed, w0):
    net = torch.load(net_path)
    TaylorNet = TaylorNetWork(order = order,  network = net)

    # Calculate the partial differential of each order
    input = torch.tensor(reference_point, dtype=torch.float)
    TaylorNet.forward(input, mixed=mixed)
    alpha_list = TaylorNet.backward('cpu')
    alpha_list = [float(alpha) for alpha in alpha_list]
    results = [abs(alpha_list[i]/alpha_list[1]) for i in range(1,len(alpha_list))]
    results = [format(results[i],'.2e') for i in range(len(results))]
    info = 'w0=' + str(w0).ljust(5,'0')
    for i in range(len(results)):
        info += ' & ' + results[i]
    print(info)

if __name__ == '__main__':
    reference_point = [[0.0]]
    order = 10
    mixed = False
    w0_list = [0.010,0.100,1.000,10.00,100.0]
    for w0 in w0_list:
        net_path = f'demo/b_convergence/outputs/Sine{w0}_1_1_512_5_True/net.pt'
        main(reference_point, order, net_path, mixed, w0)