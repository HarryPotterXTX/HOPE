import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.default'] = 'regular'
from utils.Taylor import taylor_expansion_dict, list2dict
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def create_coords(coords_shape, minimum, maximum):
    parameter = []
    for i in range(len(coords_shape)):
        parameter.append(torch.linspace(minimum[i],maximum[i],coords_shape[i]))
    coords = torch.stack(torch.meshgrid(parameter),axis=-1)
    coords = np.array(coords)
    return coords

def save_result(x, net_output, taylor_output, autograd_output, fig_path):
    fontsize = 40
    label_size = 16
    labelpad = -8
    dpi = 1000

    zlim = max(abs(net_output-taylor_output).max(), abs(net_output-autograd_output).max())

    _X, _Y = x[:,:,0], x[:,:,1]
    # plot the approximated values
    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    if zlim != 0:
        ax.set_zlim([0, zlim])
    ax.plot_surface(_X, _Y, abs(net_output-taylor_output), cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel(r'$x_1$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_ylabel(r'$x_2$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_zlabel(r'$|y-y_1|$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.tick_params(labelsize=1)
    path = os.path.join(fig_path+'_hope.png')
    plt.savefig(path)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    if zlim != 0:
        ax.set_zlim([0, zlim])
    ax.plot_surface(_X, _Y, abs(net_output-autograd_output), cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel(r'$x_1$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_ylabel(r'$x_2$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_zlabel(r'$|y-y_2|$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    path = os.path.join(fig_path+'_auto.png')
    plt.savefig(path)
    plt.close(fig)

    # z1 = max(net_output.min(), taylor_output.min(), autograd_output.min())
    # z2 = max(net_output.max(), taylor_output.max(), autograd_output.max())
    z1 = net_output.min()
    z2 = net_output.max()

    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    ax.set_zlim([z1, z2])
    ax.plot_surface(_X, _Y, net_output, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel(r'$x_1$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_ylabel(r'$x_2$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_zlabel(r'$y$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    path = os.path.join(os.path.dirname(fig_path),'origin.png')
    plt.savefig(path, dpi=dpi)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    ax.set_zlim([z1, z2])
    ax.plot_surface(_X, _Y, taylor_output, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel(r'$x_1$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_ylabel(r'$x_2$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_zlabel(r'$y_1$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    path = os.path.join(os.path.dirname(fig_path),'hope.png')
    plt.savefig(path, dpi=dpi)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    ax.set_zlim([z1, z2])
    ax.plot_surface(_X, _Y, autograd_output, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel(r'$x_1$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_ylabel(r'$x_2$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_zlabel(r'$y_2$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    path = os.path.join(os.path.dirname(fig_path),'auto.png')
    plt.savefig(path)
    plt.close(fig)

def evaluate(coords_range ,coords_length, net_path, mixed, set_order):
    # get path and work point
    net_dir = os.path.dirname(net_path)
    npy_dir = os.path.join(net_dir, 'npy')
    for path in os.listdir(npy_dir):
        if 'HOPE' in path and str(mixed) in path and str(set_order) in path:
            taylor_path = os.path.join(npy_dir, path)
        if 'Autograd' in path and str(mixed) in path and str(set_order) in path:
            auto_path = os.path.join(npy_dir, path)
    print(taylor_path, auto_path)
    work_point = [float(x) for x in taylor_path.split('[')[-1].split(']')[0].split(',')]
    fig_dir = os.path.join(net_dir, 'figs')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    fig_path = os.path.join(fig_dir, f'order{set_order}_on{work_point}_{mixed}')
    print('#'*40+' work point:{} '.format(work_point)+'#'*40 + '\n')
    coords_shape = [coords_length]*len(work_point)

    # Neural network
    net = torch.load(net_path).eval()

    # Taylor network
    taylor_list = np.load(taylor_path, allow_pickle=True)
    taylor_dict = list2dict(alpha_list=taylor_list, n_in=len(work_point), alpha_dict={}, mixed=mixed)
    
    auto_list = np.load(auto_path, allow_pickle=True)
    auto_dict = list2dict(alpha_list=auto_list, n_in=len(work_point), alpha_dict={}, mixed=mixed)

    # coords
    minimum, maximum = [point-coords_range for point in work_point], [point+coords_range for point in work_point]
    x = create_coords(coords_shape, minimum, maximum)
    origin_x = x
    origin_shape = x.shape[:-1]
    x = x.reshape((-1,x.shape[-1])) 
    
    # evaluate
    net_output = np.array(net.forward(torch.tensor(x, dtype=torch.float)).detach().numpy())   # batch point coordinate
    taylor_output = np.array(taylor_expansion_dict(taylor_dict, np.mat(x), np.array(work_point), mixed=mixed))
    auto_output = np.array(taylor_expansion_dict(auto_dict, np.mat(x), np.array(work_point), mixed=mixed))
    net_output = net_output.reshape(origin_shape)
    taylor_output = taylor_output.reshape(origin_shape)
    auto_output = auto_output.reshape(origin_shape)

    # save result
    save_result(origin_x, net_output, taylor_output, auto_output, fig_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2-D error plot')
    parser.add_argument('-d', type=str, default='demo/a_accuracy/outputs/2D_Conv_AvePool/net.pt', help='network path')
    parser.add_argument('-o', type=int, default=8, help='expansion order')
    parser.add_argument('-m', type=bool, default=True, help='mixed partial derivatives')
    parser.add_argument('-r', type=float, default=2, help='coords range')
    parser.add_argument('-l', type=int, default=30, help='coords length')
    args = parser.parse_args()
    
    net_path = args.d
    order = args.o
    mixed = args.m
    coords_range = args.r
    coords_length = args.l

    evaluate(coords_range, coords_length, net_path, mixed, set_order=order)