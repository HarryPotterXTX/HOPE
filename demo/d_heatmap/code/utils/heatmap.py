import os
import copy
import numpy as np
import cv2
import time
import argparse
import math
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../../..")))

def get_max_n(x, n):
    tempx = sorted(x.reshape(-1), reverse=True)
    x[x<tempx[n]] = 0
    return x

def normal(origin, ratio, positive=True):
    # select the first 10% positive factors and negative factors
    img = copy.deepcopy(origin)
    # print((img>0).sum()/(28*28))
    if positive == True:
        img[img<0] = 0
    else:
        img[img>0] = 0
    img = abs(img)
    img = (img-img.min())/(img.max()-img.min())
    img = (255*img)
    img = get_max_n(img, int(28*28*ratio))
    # img[img>0] = 255
    return img

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='1-D curves')
    parser.add_argument('-p', type=str, default='demo/d_heatmap/outputs/MNIST_2023_0716_141750/model/SingleOutput', help='model dir')
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

    npy_dir = os.path.join(net_dir, 'npy', f'net-{net_idx}_label-{label}_idx-{idx}')
    tay_heat = np.zeros((28,28))
    save_dir = f'demo/d_heatmap/heatmaps/net-{net_idx}_label-{label}_idx-{idx}/dx-{delta_x}_ratio-{ratio}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_start = time.time()
    for k in range(1,order+1):
        npy_path = os.path.join(npy_dir, str(k)+'.npy')
        grad = np.load(npy_path)
        tay_heat += grad/math.factorial(k)*delta_x**k
    print(f'time cost of HOPE: {time.time()-time_start}')
    np.save(os.path.join(save_dir, '1_hope'), tay_heat)
    norm_heat = (tay_heat-tay_heat.min())/(tay_heat.max()-tay_heat.min())*255
    cv2.imwrite(os.path.join(save_dir, '2_hope.png'), norm_heat)
    cv2.imwrite(os.path.join(save_dir, '3_hope_positive.png'), normal(tay_heat, ratio, True))
    cv2.imwrite(os.path.join(save_dir, '4_hope_negative.png'), normal(tay_heat, ratio, False))
        
