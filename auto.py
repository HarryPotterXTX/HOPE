import os
import sys
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm

def cald(y, x, idx):
    '''calculate ay/a(x_idx)'''
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0][:,idx-1].reshape(-1, 1)

def autograd(y:torch.tensor, x:torch.tensor, order:int=10):
    '''calculate the n-order partial derivatives with Autograd'''
    v = {0:[y]}
    p = x.numel()
    bar = order if p==1 else int((1-p**(order+1))/(1-p))-1
    pbar = tqdm(total=bar, desc='Calculating derivatives with Autograd', leave=True, file=sys.stdout)
    for k in range(1, order+1):
        v[k] = []
        for i in range(len(v[k-1])):    # eg. when p=3 and k=3, v[k-1] = [11, 12, 13, 21, 22, 23, ..., 33]
            for j in range(1, p+1):     # eg. when p=3 and k=3, v[k] = [111, 112, 113, 121, 122, 123, ..., 333]
                v[k].append(cald(v[k-1][i], x, j)) 
                pbar.set_postfix_str("order:{}, idx:{}/{}".format(k, i*p+j, len(v[k-1]*p)))
                pbar.update(1)
    return v

def main(net_path, order, point, show):
    net_dir = os.path.dirname(net_path)
    npy_dir = os.path.join(net_dir, 'npy')
    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)
    npy_path = os.path.join(npy_dir, f'Autograd_{point}_{order}').replace(" ","")

    # Forward propagation
    x = torch.tensor(point, dtype=torch.float, requires_grad=True)
    net = torch.load(net_path).eval()
    print(net)
    y = net(x)

    # Calculate all the derivatives
    time_start = time.time()
    v = autograd(y=y, x=x, order=order)
    time_auto = time.time()-time_start
    print('It takes {:.4f}s to calculate all the {}-order derivatives with Autograd.'.format(time_auto, order))
    v = {k: [float(d.detach().numpy()) for d in v[k]] for k in v.keys()}
    print(f'The number of derivatives of each order are: {str([len(v[k]) for k in sorted(v.keys())])}')
    np.save(npy_path, v)

    # Record the time cost
    time_dir = os.path.join(npy_dir, 'time')
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)
    f = open(os.path.join(time_dir,f'Autograd_{point}_{order}.txt'), 'w+')
    f.write(f'Method: Autograd \nReference Point: {point} \nOrder: {order} \nTime Cost: {time_auto}s')
    f.close()

    # show results
    if show:
        for key in range(max(v.keys())+1):
            print(f'{key} order derivatives: {v[key]}')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Taylor expansion')
    parser.add_argument('-d', type=str, default='outputs/test2d/net.pt', help='network path')
    parser.add_argument('-o', type=int, default=8, help='expansion order')
    parser.add_argument('-p', type=lambda s: [[float(item) if item[0]!='n' else -float(item[1:]) for item in s.split(',')]], default='0, 0', help='reference input')
    parser.add_argument('-s', action='store_true', help='show results')
    args = parser.parse_args()

    main(net_path=args.d, order=args.o, point=args.p, show=args.s)