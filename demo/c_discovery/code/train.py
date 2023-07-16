import torch
import os
import shutil
from tqdm import tqdm
from omegaconf import OmegaConf
import argparse
import sys
import json
import copy
import time
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from utils.Logger import MyLogger, reproduc
from utils.Network import Network
from utils.Optimizer import create_optim, create_lr_scheduler
from utils.Samplers import SamplerDict

class DeepTaylor:
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
        name = self.opt.sampler
        Sampler = SamplerDict[name]
        self.sampler = Sampler(batch_size=batch_size, epochs=epochs, device=self.device, normal_min=self.opt.Preprocess.normal_min, normal_max=self.opt.Preprocess.normal_max)
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
    Log = MyLogger(**opt['Log'])
    shutil.copy(args.p, Log.script_dir)
    shutil.copy(__file__, Log.script_dir)
    shutil.copy('utils/Samplers.py', Log.script_dir)
    reproduc()

    frame = DeepTaylor(opt.Framework, Log)
    frame.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='single task')
    parser.add_argument('-p', type=str, default='demo/c_discovery/code/exp.yaml', help='config file path')
    parser.add_argument('-g', help='availabel gpu list', default='0',
                        type=lambda s: [int(item) for item in s.split(',')])
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.g])
    main()