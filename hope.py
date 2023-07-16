import torch
import numpy as np
import os
import time
import argparse
from utils.Taylor import TaylorNetWork, list2dict

def main(work_point, order, net_path, mixed):
    net_dir = os.path.dirname(net_path)
    npy_dir = os.path.join(net_dir, 'npy')
    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)
    npy_path = os.path.join(npy_dir, 'HOPE_' + str(work_point)+'_'+str(order)+'_'+str(mixed))

    # Load the model
    net = torch.load(net_path).eval()
    TaylorNet = TaylorNetWork(order = order,  network = net)

    # Calculate the partial derivatives of each order
    time_start = time.time()
    input = torch.tensor(work_point, dtype=torch.float)
    TaylorNet.forward(input, mixed=mixed)
    alpha_list = TaylorNet.backward('cpu')
    time_taylor = time.time()-time_start
    time_dir = os.path.join(npy_dir, 'time')
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)
    f = open(os.path.join(time_dir, 'HOPE_'+str(work_point)+'_'+str(order)+'_'+str(mixed)+'.txt'),'w+')
    f.write(f'HOPE {work_point} {order} {mixed}: {time_taylor}')
    f.close()
    alpha_list = [float(alpha) for alpha in alpha_list]
    np.save(npy_path, alpha_list)

    print('time cost of HOPE: {:.4f}s'.format(time_taylor))
    print(f'the total number of derivatives is: {len(alpha_list)-1}')
    deri_info = '{:.4f}'.format(alpha_list[0])
    for i in range(min(10,len(alpha_list)-1)):
        deri_info += ', {:.2e}'.format(alpha_list[i+1])
    if len(alpha_list) > 11:
        deri_info += ' ...'
    print(f'derivatives: {deri_info}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Taylor expansion')
    parser.add_argument('-d', type=str, default='demo/1accuracy/outputs/1D_MLP_Sine/net.pt', help='network path')
    parser.add_argument('-o', type=int, default=10, help='expansion order')
    parser.add_argument('-m', type=bool, default=True, help='mixed partial derivatives')
    parser.add_argument('-p', type=lambda s: [[float(item) if item[0]!='n' else -float(item[1:]) for item in s.split(',')]], default='0', help='reference input')
    args = parser.parse_args()
    
    net_path = args.d
    order = args.o
    mixed = args.m
    work_point = args.p

    main(work_point, order, net_path, mixed)
    