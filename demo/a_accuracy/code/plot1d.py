import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.default'] = 'regular'
from utils.Taylor import taylor_expansion_dict, list2dict
os.environ['KMP_DUPLICATE_LIB_OK']='True'

colors = ["#e64b35", "#f8c370", "#5599c7", "#c39bd2", "#48c9b0", "#e6b0aa"] # red, yellow, blue

def create_coords(coords_shape, minimum, maximum):
    parameter = []
    for i in range(len(coords_shape)):
        parameter.append(torch.linspace(minimum[i],maximum[i],coords_shape[i]))
    coords = torch.stack(torch.meshgrid(parameter),axis=-1)
    coords = np.array(coords)
    return coords

def save_result(x, net_output, taylor_output, autograd_output, fig_path, work_point, point):
    width = 28/2.54
    height = 21/2.54
    fontsize = 40*1.2
    dpi = 1000
    label_size = 38*1.2
    linewidth = 3*1.5
    legendsize = 20*1.5*1
    markersize = 10*1.5
    labelpad = 9
    pad = 5

    fig = plt.figure(figsize=(width, height))
    w = [Size.Fixed(0.6), Size.Scaled(1.), Size.Fixed(.1)]
    h = [Size.Fixed(1.3), Size.Scaled(1.), Size.Fixed(.8)]
    divider = Divider(fig, (0, 0, 1, 1), w, h, aspect=False)
    ax = fig.add_axes(divider.get_position(),axes_locator=divider.new_locator(nx=1, ny=1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot(x,net_output,label='Network',c=colors[1],marker='o',linewidth=linewidth,markersize=markersize)
    if 'ReLU' in fig_path:
        ax.plot(x,autograd_output,label='Autograd',c=colors[2],marker='o',linewidth=linewidth,markersize=markersize*1.4)
    else:
        ax.plot(x,autograd_output,label='Autograd',c=colors[2],marker='o',linewidth=linewidth,markersize=markersize)
    ax.plot(x,taylor_output,label='HOPE',c=colors[0],marker='o',linewidth=linewidth,markersize=markersize)
    ax.scatter(point[0],point[1],label='Reference Input',c='black',marker='H',s=80*markersize)
    
    ax.set_xlabel('x', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_ylabel('y', fontdict={"size": fontsize}, labelpad = labelpad)
    # plt.legend(prop={"size": legendsize},loc=location)
    plt.tick_params(labelsize=label_size, pad=pad)

    if 'Sine' in fig_path and not 'Conv' in fig_path:
        plt.title('MLP with Sine',fontdict={"size": fontsize*1.2})
    elif 'ReLU' in fig_path:
        plt.title('MLP with ReLU',fontdict={"size": fontsize*1.2})
    elif 'Ave' in fig_path:
        plt.title('CNN with Average Pooling',fontdict={"size": fontsize*1.2})
    elif 'Max' in fig_path:
        plt.title('CNN with Max Pooling',fontdict={"size": fontsize*1.2})

    plt.yticks((),())    # hide yticks

    plt.savefig(fig_path+'.png', dpi=dpi)

def evaluate(coords_range ,coords_length, net_path, mixed, order, work_point):
    # get path and work point
    net_dir = os.path.dirname(net_path)
    npy_dir = os.path.join(net_dir, 'npy')
    for path in os.listdir(npy_dir):
        if 'HOPE' in path and str(mixed) in path and str(order) in path and str(work_point) in path:
            taylor_path = os.path.join(npy_dir, path)
        if 'Autograd' in path and str(mixed) in path and str(order) in path and str(work_point) in path:
            autograd_path = os.path.join(npy_dir, path)

    work_point = [float(x) for x in taylor_path.split('[')[-1].split(']')[0].split(',')]
    order = int(taylor_path.split('_')[-2])
    fig_dir = os.path.join(net_dir, 'figs')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    fig_path = os.path.join(fig_dir, f'{work_point}_{order}_{coords_range}_{mixed}')
    print('#'*40+' work point:{} '.format(work_point)+'#'*40 + '\n')
    coords_shape = [coords_length]*len(work_point)
    print(f'taylor path: {taylor_path}')
    print(f'autograd path: {autograd_path}')

    # Neural network
    net = torch.load(net_path).eval()

    # Taylor network
    taylor_list = np.load(taylor_path)
    taylor_dict = list2dict(alpha_list=taylor_list, n_in=len(work_point), alpha_dict={}, mixed=mixed)

    # Autograd 
    autograd_list = np.load(autograd_path)
    autograd_dict = list2dict(alpha_list=autograd_list, n_in=len(work_point), alpha_dict={}, mixed=mixed)
    
    print(len(taylor_dict.keys()))

    # coords
    minimum, maximum = [point-coords_range for point in work_point], [point+coords_range for point in work_point]
    x = create_coords(coords_shape, minimum, maximum)
    origin_shape = x.shape[:-1]
    x = x.reshape((-1,x.shape[-1])) 

    # evaluate
    net_output = np.array(net.forward(torch.tensor(x, dtype=torch.float)).detach().numpy())   # batch point coordinate
    taylor_output = np.array(taylor_expansion_dict(taylor_dict, np.mat(x), np.array(work_point), mixed=mixed))
    autograd_output = np.array(taylor_expansion_dict(autograd_dict, np.mat(x), np.array(work_point), mixed=mixed))

    net_output = net_output.reshape(origin_shape)
    taylor_output = taylor_output.reshape(origin_shape)
    autograd_output = autograd_output.reshape(origin_shape)

    # save result
    point = [work_point[0], taylor_list[0]]
    save_result(x, net_output, taylor_output, autograd_output, fig_path, work_point, point)

if __name__ == '__main__':
    global location
    location = 'lower center'

    parser = argparse.ArgumentParser(description='Taylor expansion')
    parser.add_argument('-d', type=str, default='demo/a_accuracy/outputs/1D_MLP_Sine/net.pt', help='network path')
    parser.add_argument('-o', type=int, default=10, help='expansion order')
    parser.add_argument('-m', type=bool, default=True, help='mixed partial derivatives')
    parser.add_argument('-p', type=lambda s: [[float(item) for item in s.split(',')]], default='0', help='reference input')
    parser.add_argument('-r', type=float, default=5, help='coords range')
    parser.add_argument('-l', type=int, default=30, help='coords length')
    args = parser.parse_args()
    
    net_path = args.d
    order = args.o
    mixed = args.m
    work_point = args.p
    coords_range = args.r
    coords_length = args.l

    evaluate(coords_range, coords_length, net_path, mixed, order, work_point)