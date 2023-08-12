import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../../..")))
from utils.Network import MLP, CONV
from utils.Logger import reproduc

if __name__=='__main__':
    reproduc()

    # 2-D CNN
    net = CONV(input=2, in_channel=2, out_channel=16, height=32, 
               width=32, pool='Ave', act='Sine')
    save_path = os.path.join('demo/a_accuracy/outputs',f'2D_Conv_AvePool','net.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(net,save_path)