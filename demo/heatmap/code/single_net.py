import torch
import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from demo.heatmap.code.trainMNIST import MNISTNet

class SingleNet(torch.nn.Module):
    def __init__(self, orig_net, out_idx):
        super().__init__()
        self.idx = out_idx
        self.net = torch.nn.Sequential()
        for i in range(len(orig_net.net)):
            module = orig_net.net[i]
            # if 'Dropout2d' in str(type(module)):
            #     continue
            if i == len(orig_net.net)-1:
                last_module = torch.nn.Linear(module.weight.shape[1], 1)
                with torch.no_grad():
                    last_module.weight.copy_(module.weight[out_idx:out_idx+1])
                    last_module.bias.copy_(module.bias[out_idx:out_idx+1])
                module = last_module
            self.net.append(module)

    def forward(self,input):
        output = self.net(input)
        return output

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='project dir')
    parser.add_argument('-p', type=str, default='demo/heatmap/outputs/MNIST_2023_1009_082528', help='model dir')
    args = parser.parse_args()
    pro_dir = args.p
    path = os.path.join(pro_dir, 'model', 'best.pt')
    save_dir = os.path.join(os.path.dirname(path), 'SingleOutput')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    net = torch.load(path).cpu().eval()
    net.net = net.net.eval()
    SingleNetworks = [SingleNet(net, idx).cpu().eval() for idx in range(10)]
    for idx in range(len(SingleNetworks)):
        torch.save(SingleNetworks[idx], os.path.join(save_dir, str(idx) + '.pt'))