import torch
import numpy as np
import os
import time
import argparse
# import sys
# sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

def cald(y,x,idx):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0][:, idx].reshape(-1, 1)

def Autograd_Expansion(work_point, order, net_path, mixed):
    net_dir = os.path.dirname(net_path)
    auto_dir = os.path.join(net_dir, 'npy')
    if not os.path.exists(auto_dir):
        os.mkdir(auto_dir)
    save_path = os.path.join(auto_dir, 'Autograd_'+str(work_point)+'_'+str(order)+'_'+str(mixed))

    net = torch.load(net_path).eval()
    net.requires_grad = False
    x = torch.tensor(work_point, dtype=torch.float32, requires_grad=True)
    y = net(x)

    info_dict = {0:[y]}
    info_list = [y]
    time_start = time.time()
    p = len(work_point[0])
    for i in range(1,order+1):
        info_dict[i]=[]
        for j in range(len(info_dict[i-1])):
            y = info_dict[i-1][j]
            for k in range(p):
                if not mixed:
                    if len(info_dict[i-1])==1:
                        d = cald(y,x,k)
                        info_dict[i].append(d)
                    elif j==k:
                        d = cald(y,x,k)
                        info_dict[i].append(d)
                else:
                    d = cald(y,x,k)
                    info_dict[i].append(d)
                    # print(i,j,k)
        info_list += info_dict[i]
        if p > 1:
            print('Autograd {} order: {:.6f}s'.format(i, time.time()-time_start))
    time_auto = time.time()-time_start
    info_list = [float(d.detach().numpy()) for d in info_list]
    np.save(save_path, info_list)

    time_dir = os.path.join(auto_dir, 'time')
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)
    f = open(os.path.join(time_dir,'Autograd_'+str(work_point)+'_'+str(order)+'_'+str(mixed)+'.txt'),'w+')
    f.write(f'auto {work_point} {order} {mixed}: {time_auto}')
    f.close()

    print('time cost of Autograd: {:.4f}s'.format(time_auto))
    print(f'the total number of derivatives is: {len(info_list)-1}')
    deri_info = '{:.4f}'.format(info_list[0])
    for i in range(min(10,len(info_list)-1)):
        deri_info += ', {:.2e}'.format(info_list[i+1])
    if len(info_list) > 11:
        deri_info += ' ...'
    print(f'derivatives: {deri_info}')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Autograd expansion')
    parser.add_argument('-d', type=str, default='demo/1accuracy/outputs/1D_MLP_Sine/net.pt', help='network path')
    parser.add_argument('-o', type=int, default=10, help='expansion order')
    parser.add_argument('-m', type=bool, default=True, help='mixed partial derivatives')
    parser.add_argument('-p', type=lambda s: [[float(item) for item in s.split(',')]], default='0', help='reference input')
    args = parser.parse_args()
    
    net_path = args.d
    order = args.o
    mixed = args.m
    work_point = args.p

    Autograd_Expansion(work_point, order, net_path, mixed)