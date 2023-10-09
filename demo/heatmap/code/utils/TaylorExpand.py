import torch
import numpy as np
import os
import time
import cv2
from einops import rearrange
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../../..")))
from utils.Logger import reproduc
from demo.heatmap.code.utils.heatmap import normal
from demo.heatmap.code.trainMNIST import MNISTNet
from demo.heatmap.code.single_net import SingleNet 
from utils.HopeGrad import hopegrad

def main(input, order, net_dir, mixed, net_idx):
    net_path = os.path.join(net_dir, str(net_idx) + '.pt')
    taylor_dir = os.path.join(net_dir, 'npy', f'net-{net_idx}_label-{label}_idx-{idx}')
    if not os.path.exists(taylor_dir):
        os.makedirs(taylor_dir)

    # Load the model
    net = torch.load(net_path).cpu().eval()

    # Forward propagation
    input = torch.tensor(input, dtype=torch.float, requires_grad=True)
    output = net(input)
    print(output)

    # Calculate the partial differential of each order
    time_start = time.time()
    hopegrad(y=output, order=order, mixed=0)    # mixed=0: calculate all the unmixed partial derivatives
    print(f'time cost: {time.time()-time_start}')
    v = input.hope_grad

    for i in range(1,order+1):
        poly_path = os.path.join(taylor_dir, str(i))
        heat = v[i].detach().numpy().reshape(28,28)
        np.save(poly_path, heat)
        cv2.imwrite(poly_path+'.png', normal(heat, 0.1))

if __name__ == '__main__':
    global label, idx
    parser = argparse.ArgumentParser(description='1-D curves')
    parser.add_argument('-p', type=str, default='demo/heatmap/outputs/MNIST_2023_1009_082528/model/SingleOutput', help='model dir')
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
    img_path = f'demo/heatmap/data/test/{label}/{idx}.png'
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    input = torch.tensor([[image]])
    mixed = False
    main(input, order, net_dir, mixed, net_idx)