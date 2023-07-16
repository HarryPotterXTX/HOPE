import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='1-D curves')
    parser.add_argument('-p', type=str, default='demo/d_heatmap/outputs/MNIST_2023_0716_141750', help='model dir')
    args = parser.parse_args()
    pro_dir = args.p

    order = 10
    label, idx = 0, 0
    delta_x, ratio = 1, 0.1
    for net_idx in range(10):
        net_dir = os.path.join(pro_dir, 'model', 'SingleOutput')
        npy_dir = os.path.join(net_dir, 'npy', f'net-{net_idx}_label-{label}_idx-{idx}')
        if not os.path.exists(npy_dir):
            os.system(f'python demo/d_heatmap/code/utils/TaylorExpand.py -p {net_dir} -n {net_idx} -l {label} -i {idx} -o {order}')
        os.system(f'python demo/d_heatmap/code/utils/perturbation.py -p {net_dir} -n {net_idx} -l {label} -i {idx} -o {order} -d {delta_x} -r {ratio}')
        os.system(f'python demo/d_heatmap/code/utils/heatmap.py -p {net_dir} -n {net_idx} -l {label} -i {idx} -o {order} -d {delta_x} -r {ratio}')