import torch
import os
import copy
import numpy as np
import cv2
import time
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../../..")))
from demo.heatmap.code.utils.heatmap import normal
from demo.heatmap.code.trainMNIST import MNISTNet
from demo.heatmap.code.single_net import SingleNet 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='1-D curves')
    parser.add_argument('-p', type=str, default='demo/heatmap/outputs/MNIST_2023_0716_141750/model/SingleOutput', help='model dir')
    parser.add_argument('-n', type=int, default=0, help='model idx')
    parser.add_argument('-l', type=int, default=0, help='label of img')
    parser.add_argument('-i', type=int, default=0, help='img index')
    parser.add_argument('-o', type=int, default=10, help='expansion order')
    parser.add_argument('-d', type=int, default=1, help='delta x')
    parser.add_argument('-r', type=float, default=0.1, help='display ratio')
    args = parser.parse_args()
    label, idx = args.l, args.i
    net_idx = args.n
    order = args.o
    delta_x = args.d
    ratio = args.r
    net_dir = args.p

    net = torch.load(os.path.join(net_dir, str(net_idx)+'.pt')).cpu().eval()
    save_dir = f'demo/heatmap/heatmaps/net-{net_idx}_label-{label}_idx-{idx}/dx-{delta_x}_ratio-{ratio}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img_path = f'demo/heatmap/data/test/{label}/{idx}.png'
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    x0 = torch.tensor([[image]],dtype=torch.float32)
    y0 = net(x0)
    heat = np.zeros((28,28))
    time_start = time.time()
    for i in range(28):
        for j in range(28):
            x = copy.deepcopy(x0)
            x[:,:,i,j] += delta_x
            y = net(x)
            delta_y = y - y0
            heat[i,j] = delta_y
    print(f'time cost of perturbation: {time.time()-time_start}')
    cv2.imwrite(os.path.join(save_dir, '0_origin.png'), image)
    np.save(os.path.join(save_dir, '1_net'), heat)
    norm_heat = (heat-heat.min())/(heat.max()-heat.min())*255
    cv2.imwrite(os.path.join(save_dir, '2_net.png'), norm_heat)
    cv2.imwrite(os.path.join(save_dir, '3_net_positive.png'), normal(heat, ratio, True))
    cv2.imwrite(os.path.join(save_dir, '4_net_negative.png'), normal(heat, ratio, False))
        
