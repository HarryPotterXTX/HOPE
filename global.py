import os
import math
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams["font.family"] = "Times New Roman"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utils.HopeGrad import hopegrad
from utils.Samplers import create_flattened_coords
from utils.Global import convert_derivatives, create_coords

def get_interactions(p:int, n:int):
    labels = ['c']
    for k in range(1,n+1):
        idxs = create_coords([p]*k, [1]*k, [p]*k).astype(np.int16)
        idxs = idxs.reshape((-1, idxs.shape[-1]))
        for idx in idxs:
            label = r''
            for id in idx:
                label += '$x_{}$\n'.format(id)
            labels.append(label)
    return labels

def get_taylor_coef(grad_dict:dict, order:int):    
    '''return [y0,v1/1!,v2/2!,...,vn/n!]'''
    coef = [grad_dict[0]]
    for k in range(1,order+1):
        coef.append(np.array(grad_dict[k])/math.factorial(k))
    coef = np.concatenate(coef)
    return coef

def judge_exist(net_dir, order, x):
    '''search data available for direct use'''   
    for n in range(order, order+10):
        npy_path = os.path.join(net_dir, 'npy', f'HOPE_[[{x}]]_{n}.npy').replace(" ","")
        if os.path.exists(npy_path):
            return 1, npy_path
    return 0, os.path.join(net_dir, 'npy', f'HOPE_[[{x}]]_{order}.npy').replace(" ","")

def plot(coefs, ref_coef, title, xlabels, order, global_dir, save):
    BOX_EDGE_COLOR = "#3A3A3A"
    BOX_COLOR = ["#d98880", "#76d7c3", "#f9d7a0", "#7fb3d5", "#e6b0aa", "#d5dadb"]
    def deal_zero(num:float):
        return 0.00 if '{:.2f}'.format(num)=='-0.00' else num 
    vmax, vmin = max(coefs.max(), ref_coef.max()), min(coefs.min(), ref_coef.min())
    var_num = ref_coef.shape[0]
    w, h = (6.4/12)*(1.5+0.5+var_num), 4.8+order*(4.8*0.05)    # w: 1.5+0.5+n; h: 
    plt.figure(figsize=(w, h))
    for idx in range(coefs.shape[1]):
        plt.boxplot(coefs[:,idx],
            positions=[50+100*idx], widths=30,
            whis=1.5, patch_artist=True,
            boxprops=dict(linestyle='-', linewidth=1, color=BOX_EDGE_COLOR, facecolor=BOX_COLOR[idx%len(BOX_COLOR)]),
            capprops=dict(linestyle='-', linewidth=1, color=BOX_EDGE_COLOR),
            whiskerprops=dict(linestyle='-', linewidth=1, color=BOX_EDGE_COLOR),
            medianprops=dict(linestyle='-', linewidth=1, color=BOX_EDGE_COLOR),
            flierprops=dict(marker='.', markerfacecolor=BOX_EDGE_COLOR, markersize=1, markeredgecolor=BOX_EDGE_COLOR))
        plt.plot(50+100*idx, ref_coef[idx], color='red', marker='o', markersize=6)
        y = (coefs[:,idx].mean()+ref_coef[idx])/2
        y1 = (vmax-vmin)*0.04+y if coefs[:,idx].mean()>ref_coef[idx] else -(vmax-vmin)*0.04+y
        y2 = -(vmax-vmin)*0.04+y if coefs[:,idx].mean()>ref_coef[idx] else (vmax-vmin)*0.04+y
        plt.text(50+100*idx+20, y1, '{:.2f}'.format(deal_zero(coefs[:,idx].mean())), color='blue')
        plt.text(50+100*idx+20, y2, '{:.2f}'.format(deal_zero(ref_coef[idx])), color='red')
        if xlabels[idx]=='c':
            plt.text(50+100*idx+20, y, '{:.2f}'.format(deal_zero(ref_coef[idx])), color='black')
        else:
            plt.text(50+100*idx+20, y, '{:.2f}'.format(deal_zero((coefs[:,idx].mean()+ref_coef[idx])/2)), color='black')
    ax = plt.gca()
    ax.set_xticklabels(xlabels, fontsize=12) 
    plt.xlim(0,100*idx+150)
    plt.title(title, fontsize=20)
    # plt.xlabel('Feature Interactions', fontsize=18) 
    plt.ylabel('Taylor Coefficients', fontsize=18) 
    plt.subplots_adjust(left=1.5/(1.5+0.5+var_num), right=0.99, top=0.93, bottom=(order*(4.8*0.05))/(4.8+order*(4.8*0.05)))
    if not save:
        plt.show()
    elif 'Top' in title:
        plt.savefig(os.path.join(global_dir,'global_top.png'))
    else:
        plt.savefig(os.path.join(global_dir,'global.png'))

def main(net_path, order, point, num, coord_range, top, save):
    net = torch.load(net_path).eval()
    print(net)

    # Taylor coefficients obtained at multiple points
    shape = [[point[i]-coord_range,point[i]+coord_range,num] for i in range(len(point))]
    coords = create_flattened_coords(shape) 
    coords = torch.concat([coords, torch.tensor([point])], dim=0)
    coords.requires_grad = True  
    batch, p = coords.shape
    outputs = net(coords)
    # Back-propagation
    hopegrad(y=outputs, order=order, mixed=1)
    hope_grad_dict = {k:coords.hope_grad[k].detach().numpy().reshape(coords.hope_grad[k].shape[0],-1) for k in coords.hope_grad.keys()}
    outputs = outputs.detach().numpy()
    coords = coords.detach().numpy()
    coefs = np.zeros((batch, order+1)) if p==1 else np.zeros((batch, 1+int(p*(1-p**order)/(1-p))))
    for i in range(coords.shape[0]):
        # Get the Taylor polynomial at point x0
        grad0_dict = {k:hope_grad_dict[k][i] for k in hope_grad_dict.keys()}   
        grad0_dict[0] = outputs[i]
        # Convert the Taylor polynomial obtained at x0 to point x1
        grad1_dict = convert_derivatives(grad_dict=grad0_dict, x0=coords[i:i+1], x1=np.array(point))    
        # Calculate Taylor coefficients with all the derivatives
        coef = get_taylor_coef(grad1_dict, order)
        coefs[i] = coef

    # Taylor coefficient on refenrence points
    ref_coef = coef

    # png path
    global_dir = os.path.join(os.path.dirname(net_path), 'global')
    if save and not os.path.exists(global_dir):
        os.mkdir(global_dir)

    # show the results
    plot(coefs, ref_coef, title=f'Coefficients on {point}', xlabels=get_interactions(p, order), order=order, global_dir=global_dir, save=save)

    # show the results of the top m
    top_ref_coef = ref_coef[abs(ref_coef).argsort()[-top:]]
    top_mean_coef = coefs[:,abs(ref_coef).argsort()[-top:]]
    xlabels = [get_interactions(p, order)[i] for i in abs(ref_coef).argsort()[-top:]]
    plot(top_mean_coef, top_ref_coef, title=f'Top {top} coefficients on {point}', xlabels=xlabels, order=order, global_dir=global_dir, save=save)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Taylor expansion on multiple points')
    parser.add_argument('-d', type=str, default='demo/discovery/outputs/discovery_2024_0202_181236/model/best.pt', help='network path')
    parser.add_argument('-o', type=int, default=4, help='expansion order')
    parser.add_argument('-p', type=lambda s: [float(item) if item[0]!='n' else -float(item[1:]) for item in s.split(',')], default='0, 0', help='reference input')
    parser.add_argument('-n', type=int, default=10, help='number of samples')
    parser.add_argument('-r', type=float, default=1, help='coords range')
    parser.add_argument('-t', type=int, default=10, help='show the results of the top t')
    parser.add_argument('--save', default=False, action='store_true', help='save png')
    args = parser.parse_args()

    main(net_path=args.d, order=args.o, point=args.p, num=args.n, coord_range=args.r, top=args.t, save=args.save)