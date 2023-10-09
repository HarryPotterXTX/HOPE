import copy
import torch

def create_optim(name, parameters ,lr):
    if name == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif name == 'Adamax':
        optimizer = torch.optim.Adamax(parameters, lr=lr)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=lr)
    else:
        raise NotImplemented
    return optimizer

def create_lr_scheduler(optimizer, lr_scheduler_opt):
    lr_scheduler_opt = copy.deepcopy(lr_scheduler_opt)
    lr_scheduler_name = lr_scheduler_opt.pop('name')
    if lr_scheduler_name == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'CyclicLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'none':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000000000])
    else:
        raise NotImplementedError
    return lr_scheduler