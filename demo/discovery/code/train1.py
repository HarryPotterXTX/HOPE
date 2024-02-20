import os
import sys
import json
import copy
import time
import torch
import shutil
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from utils.Logger import MyLogger, reproduc
from utils.Network import Network
from utils.Optimizer import create_optim, create_lr_scheduler
from utils.Samplers import create_flattened_coords, normalize_data, BaseSampler

# y=f(x1, x2, x3, x4, z)=0.5+x1+0.6x2x3-0.8*x2x4-zx3+x1x2x3, x in [-1,1]
# mode=1: x1~x4->net(ignore z)->y; mode=2: x1~x4->net(know z)->y; mode=3: x1~x4,z->net->y
class Sampler(BaseSampler):
    def __init__(self, batch_size: int, epochs:int, device:str='cpu', normal_min:float=-1, normal_max:float=1, mode:int=1, z_value:float=0) -> None:
        super().__init__(batch_size=batch_size, epochs=epochs, device=device)
        self.mode = mode
        self.z_value = z_value
        r1, r2 = -1, 1
        num = 32
        if self.mode == 1 or self.mode == 3:
            self.shape = [[r1,r2,num], [r1,r2,num], [r1,r2,num], [r1,r2,num], [r1,r2,num]]
        elif self.mode == 2:
            self.shape = [[r1,r2,num], [r1,r2,num], [r1,r2,num], [r1,r2,num], [self.z_value,self.z_value,1]]
        self.coords = create_flattened_coords(self.shape)
        [x1, x2, x3, x4, z] = [self.coords[:,i:i+1] for i in range(5)]
        # self.label = x1+x2*x3-x2*x4-z*x3
        self.label = 0.5+x1+0.6*x2*x3-0.8*x2*x4-z*x3+x1*x2*x3
        self.label, self.side_info = normalize_data(self.label, scale_min=normal_min, scale_max=normal_max)
        self.label = self.label.to(device)
        self.coords = self.coords[:,:-1].to(device) if self.mode!=3 else self.coords.to(device)
        self.pop_size = self.label.shape[0]
        
    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            sampled_coords = self.coords[sampled_idxs, :]
            sampled_label = self.label[sampled_idxs, :]
            self.index += self.batch_size
            return sampled_coords, sampled_label
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration

class DiscoveryFramework:
    def __init__(self, opt, Log) -> None:
        self.opt = opt
        self.Log = Log
        self.device = opt.Train.device
        if os.path.exists(self.opt.Network.pretrained):
            print('Load pretrained model: {}'.format(self.opt.Network.pretrained))
            self.net = torch.load(self.opt.Network.pretrained).to(self.device)
        else:
            self.net = Network(self.opt.Network.structure).to(self.device)
        self.best_net = copy.deepcopy(self.net)
        self.step_net = copy.deepcopy(self.net)
        self.sampler = self.init_sampler()
        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_lr_scheduler()
    
    def init_sampler(self):
        batch_size = self.opt.Train.batch_size
        epochs = self.opt.Train.epochs
        self.sampler = Sampler(batch_size=batch_size, epochs=epochs, device=self.device, normal_min=self.opt.Preprocess.normal_min,
            normal_max=self.opt.Preprocess.normal_max, mode=self.opt.sampler.mode, z_value=self.opt.sampler.z)
        self.side_info = self.sampler.side_info
        with open(os.path.join(self.Log.model_dir,'side_info.json'), 'w+') as f:
            json.dump(self.side_info, f)
        f.close()
        return self.sampler
    
    def init_optimizer(self):
        name = self.opt.Train.optimizer.type
        lr = self.opt.Train.optimizer.lr
        parameters = [{'params':self.net.parameters()}]
        self.optimizer = create_optim(name, parameters, lr)
        return self.optimizer
    
    def init_lr_scheduler(self):
        self.lr_scheduler = create_lr_scheduler(self.optimizer, self.opt.Train.lr_scheduler)
        return self.lr_scheduler

    def train(self):
        pbar = tqdm(self.sampler, desc='Training', leave=True, file=sys.stdout)
        mean_loss = 0
        for step, (sampled_x, sampled_y) in enumerate(pbar): 
            self.optimizer.zero_grad()
            y_hat = self.net(sampled_x)
            loss = torch.mean((y_hat - sampled_y)**2)        
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
            pbar.update(1)
            mean_loss += loss.item()
            loss_best = 1000000
            if self.sampler.judge_eval(self.opt.Eval.epochs):
                mean_loss /= self.opt.Eval.epochs
                self.Log.log_metrics({'mean_loss':mean_loss}, self.sampler.epochs_count)
                if mean_loss < loss_best:
                    loss_best = mean_loss
                    self.best_net = copy.deepcopy(self.net)
                    torch.save(self.best_net.cpu(), os.path.join(self.Log.model_dir, 'best.pt'))
                mean_loss = 0

        torch.save(self.net.cpu(), os.path.join(self.Log.model_dir, 'final.pt'))
        torch.save(self.best_net.cpu(), os.path.join(self.Log.model_dir, 'best.pt'))
        self.Log.log_metrics({'time':time.time()-time_start}, 0)
        self.Log.close()
    
def main():
    global time_start
    time_start = time.time()
    opt = OmegaConf.load(args.p)
    opt.Log.project_name = opt.Log.project_name + '_' + str(opt.Framework.sampler.mode)
    if opt.Framework.sampler.mode == 2:
        opt.Log.project_name += '_' + str(opt.Framework.sampler.z)
    if opt.Framework.sampler.mode == 3:
        opt.Framework.Network.structure[0] = 'Linear_5_64'
    Log = MyLogger(**opt['Log'])
    shutil.copy(args.p, Log.script_dir)
    shutil.copy(__file__, Log.script_dir)
    # shutil.copy('utils/Samplers.py', Log.script_dir)
    reproduc()

    frame = DiscoveryFramework(opt.Framework, Log)
    frame.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='single task')
    parser.add_argument('-p', type=str, default='demo/discovery/code/exp1.yaml', help='config file path')
    parser.add_argument('-g', help='availabel gpu list', default='0',
                        type=lambda s: [int(item) for item in s.split(',')])
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.g])
    main()