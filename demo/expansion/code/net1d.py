import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from utils.Network import MLP, CONV
from utils.Logger import reproduc

if __name__=='__main__':
    reproduc()

    # 1-D MLP with Sine
    net = MLP(input=1, hidden=1024, output=1, layer=5, act='Sine', output_act=True)
    save_path = os.path.join('demo','expansion','outputs', '1D_MLP_Sine', 'net.pt') 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(net,save_path)