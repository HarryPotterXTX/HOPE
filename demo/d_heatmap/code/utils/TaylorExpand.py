import torch
import numpy as np
import os
import time
import cv2
from einops import rearrange
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../../..")))
from utils.Taylor import TaylorNetWork
from utils.Logger import reproduc
from demo.d_heatmap.code.utils.heatmap import normal
from demo.d_heatmap.code.trainMNIST import MNISTNet
from demo.d_heatmap.code.single_net import SingleNet 

def main(input, order, net_dir, mixed, net_idx):
    net_path = os.path.join(net_dir, str(net_idx) + '.pt')
    taylor_dir = os.path.join(net_dir, 'npy', f'net-{net_idx}_label-{label}_idx-{idx}')
    if not os.path.exists(taylor_dir):
        os.makedirs(taylor_dir)

    # Load the model
    net = torch.load(net_path).cpu().eval()
    TaylorNet = TaylorNetWork(order = order,  network = net)

    # Calculate the partial differential of each order
    time1 = time.time()
    input = torch.tensor(input, dtype=torch.float)
    TaylorNet.forward(input, mixed=mixed)
    alpha_list = TaylorNet.backward('cpu')
    print(f'time cost: {time.time()-time1}')
    alpha_list = [float(alpha) for alpha in alpha_list]
    print(f'alpha:{alpha_list[:11]}...')

    output = alpha_list[0]
    print(output)

    for i in range(1,order+1):
        taylor_path = os.path.join(taylor_dir, str(i))
        heat = np.array(alpha_list[28*28*(i-1)+1:28*28*i+1]).reshape(28,28)
        np.save(taylor_path, heat)
        cv2.imwrite(taylor_path+'.png', normal(heat, 0.1))

if __name__ == '__main__':
    global label, idx
    parser = argparse.ArgumentParser(description='1-D curves')
    parser.add_argument('-p', type=str, default='demo/d_heatmap/outputs/MNIST_2023_0716_141750/model/SingleOutput', help='model dir')
    parser.add_argument('-n', type=int, default=0, help='model idx')
    parser.add_argument('-l', type=int, default=0, help='label of img')
    parser.add_argument('-i', type=int, default=0, help='img index')
    parser.add_argument('-o', type=int, default=10, help='expansion order')
    args = parser.parse_args()
    label, idx = args.l, args.i
    net_idx = args.n
    order = args.o
    net_dir = args.p

    reproduc()
    img_path = f'demo/d_heatmap/data/test/{label}/{idx}.png'
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    input = torch.tensor([[image]])
    mixed = False
    main(input, order, net_dir, mixed, net_idx)