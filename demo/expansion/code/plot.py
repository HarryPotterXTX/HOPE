import os
import math
import torch
import argparse
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import Divider, Size
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams["font.family"] = "Times New Roman"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
colors = ["#e64b35", "#f8c370", "#5599c7", "#c39bd2", "#48c9b0", "#e6b0aa"] # red, yellow, blue

from utils.Global import create_coords, taylor_output

def R2(y, y1):
    y, y1 = y.flatten(), y1.flatten()
    mse = ((y - y1)**2).sum()
    var = ((y - y.mean())**2).sum()
    return 1-mse/var

def plot1d(x, y_net, y_poly, fig_path, point):
    width, height = 28/2.54, 21/2.54
    fontsize, label_size = 40*1.2, 38*1.2
    linewidth, legendsize, markersize = 3*1.5, 20*1.5*1, 10*1.5
    labelpad, pad = 9, 5

    fig = plt.figure(figsize=(width, height))
    w = [Size.Fixed(0.6), Size.Scaled(1.), Size.Fixed(.1)]
    h = [Size.Fixed(1.3), Size.Scaled(1.), Size.Fixed(.9)]
    divider = Divider(fig, (0, 0, 1, 1), w, h, aspect=False)
    ax = fig.add_axes(divider.get_position(),axes_locator=divider.new_locator(nx=1, ny=1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot(x,y_net,label='Network',c=colors[1],marker='o',linewidth=linewidth,markersize=markersize*1.4)
    ax.plot(x,y_poly,label='Polynomial',c=colors[0],marker='o',linewidth=linewidth,markersize=markersize)
    ax.scatter(point[0],point[1],label='Reference Input',c='black',marker='H',s=80*markersize)

    ax.set_xlabel('x', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_ylabel('y', fontdict={"size": fontsize}, labelpad = labelpad)
    plt.title('Taylor Polynomial ($R^2=$'+'{:.4f})'.format(R2(y=y_net,y1=y_poly)),fontdict={"size": fontsize*1.0})
    plt.legend(prop={"size": legendsize},loc='best')
    plt.tick_params(labelsize=label_size, pad=pad)
    plt.yticks((),())
    plt.savefig(fig_path)

def plot2d(x, y_net, y_poly, fig_path):
    fontsize, labelpad = 40, -8

    _X, _Y = x[:,:,0], x[:,:,1]
    z1, z2 = y_net.min(), y_net.max()

    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    ax.set_zlim([z1, z2])
    ax.plot_surface(_X, _Y, y_net, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel(r'$x_1$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_ylabel(r'$x_2$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_zlabel(r'$y$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    path = os.path.join(os.path.dirname(fig_path),'network.png')
    plt.savefig(path)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    ax.set_zlim([z1, z2])
    ax.plot_surface(_X, _Y, y_poly, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel(r'$x_1$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_ylabel(r'$x_2$', fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_zlabel(r'$y_1$ ($R^2$='+'{:.4f})'.format(R2(y=y_net,y1=y_poly)), fontdict={"size": fontsize}, labelpad = labelpad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(fig_path)
    plt.close(fig)

def hope_output(y0, dx, grads, idx, order):
    y = y0
    for k in range(1,order+1):
        y = y + grads[k-1]/math.factorial(k)*(dx**k)
    return y

def evaluate(npy_path, pro_dir, coord_range):
    # fig path
    fig_dir = os.path.join(pro_dir, 'figs')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    fig_name = os.path.basename(npy_path)[:-4] + '.png'
    fig_path = os.path.join(fig_dir, fig_name)

    # load models
    net_path = os.path.join(pro_dir, 'net.pt')
    net = torch.load(net_path).eval()

    # load derivatives
    grad_dict = np.load(npy_path, allow_pickle=True).item()

    # coords
    point = [float(x) for x in fig_name.split('[')[-1].split(']')[0].split(',')]
    x = create_coords([30]*len(point), [x-coord_range for x in point], [x+coord_range for x in point]) 
    origin_shape = x.shape[:-1]
    xf = x.reshape((-1, x.shape[-1])) 
    dx = create_coords([30]*len(point), [-coord_range]*len(point), [coord_range]*len(point))
    dx = dx.reshape((-1, dx.shape[-1])) 

    # evaluate
    net_output = np.array(net.forward(torch.tensor(xf, dtype=torch.float)).detach().numpy()).reshape(origin_shape)
    poly_output = taylor_output(grad_dict=grad_dict, dx=dx).reshape(origin_shape)

    # save result
    if len(point) == 1:
        plot1d(x, net_output, poly_output, fig_path, (point, grad_dict[0]))
    elif len(point) == 2:
        plot2d(x, net_output, poly_output, fig_path)
    else: 
        raise NotImplemented

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='expansion plot')
    parser.add_argument('-d', type=str, default='outputs/test2d/npy/Autograd_[[0.0, 0.0]]_8.npy', help='derivative path')
    parser.add_argument('-r', type=float, default=2, help='coords range')
    args = parser.parse_args()

    npy_path = args.d
    pro_dir = os.path.dirname(os.path.dirname(npy_path))
    coord_range = args.r

    evaluate(npy_path, pro_dir, coord_range)