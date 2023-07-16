import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Expansion')
    parser.add_argument('-d', type=str, default='demo/a_accuracy/outputs/1D_MLP_Sine/net.pt', help='network path')
    parser.add_argument('-o', type=int, default=10, help='expansion order')
    parser.add_argument('-m', type=bool, default=True, help='mixed partial derivatives')
    parser.add_argument('-p', type=str, default='0', help='work point')
    parser.add_argument('-r', type=float, default=5, help='coords range')
    parser.add_argument('-l', type=int, default=30, help='coords length')
    args = parser.parse_args()

    work_point = [[float(item) for item in args.p.split(',')]]
    hope_path = os.path.join(os.path.dirname(args.d), 'npy', f'HOPE_{work_point}_{args.o}_True.npy')
    auto_path = os.path.join(os.path.dirname(args.d), 'npy', f'Autograd_{work_point}_{args.o}_True.npy')
    if not os.path.exists(hope_path):
        os.system(f'python hope.py -d {args.d} -o {args.o} -p {args.p}')
    if not os.path.exists(auto_path):
        os.system(f'python auto.py -d {args.d} -o {args.o} -p {args.p}')

    if len(args.p.split(',')) == 1:
        os.system(f'python demo/a_accuracy/code/plot1d.py -d {args.d} -m {args.m} -r {args.r} -l {args.l} -o {args.o} -p {args.p}')
    elif len(args.p.split(',')) == 2:
        os.system(f'python demo/a_accuracy/code/plot2d.py -d {args.d} -o {args.o} -m {args.m} -r {args.r} -l {args.l}')
    else:
        pass