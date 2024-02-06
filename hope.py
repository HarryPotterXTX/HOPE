import os
import time
import torch
import argparse
import numpy as np

from utils.HopeGrad import hopegrad

def main(net_path, order, point, show):
    net_dir = os.path.dirname(net_path)
    npy_dir = os.path.join(net_dir, 'npy')
    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)
    npy_path = os.path.join(npy_dir, f'HOPE_{point}_{order}').replace(" ","")

    # Forward propagation
    x = torch.tensor(point, dtype=torch.float, requires_grad=True)
    net = torch.load(net_path).eval()
    print(net)
    y = net(x)

    # Calculate all the derivatives
    time_start = time.time()
    hopegrad(y=y, order=order, mixed=1) # mixed=1: calculate all the mixed partial derivatives
    time_hope = time.time()-time_start
    print('It takes {:.4f}s to calculate all the {}-order derivatives with HOPE.'.format(time_hope, order))
    v = {k: list(x.hope_grad[k].detach().reshape(-1).numpy()) for k in x.hope_grad.keys()}
    v[0] = [float(y.detach().numpy()[0,0])]
    print(f'The number of derivatives of each order are: {str([len(v[k]) for k in sorted(v.keys())])}')
    np.save(npy_path, v)

    # Record the time cost
    time_dir = os.path.join(npy_dir, 'time')
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)
    f = open(os.path.join(time_dir,f'HOPE_{point}_{order}.txt'), 'w+')
    f.write(f'Method: HOPE \nReference Point: {point} \nOrder: {order} \nTime Cost: {time_hope}s')
    f.close()

    # show results
    if show:
        for key in range(max(v.keys())+1):
            print(f'{key} order derivatives: {v[key]}')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Taylor expansion')
    parser.add_argument('-d', type=str, default='outputs/test2d/net.pt', help='network path')
    parser.add_argument('-o', type=int, default=10, help='expansion order')
    parser.add_argument('-p', type=lambda s: [[float(item) if item[0]!='n' else -float(item[1:]) for item in s.split(',')]], default='0, 0', help='reference input')
    parser.add_argument('-s', action='store_true', help='show results')
    args = parser.parse_args()

    main(net_path=args.d, order=args.o, point=args.p, show=args.s)