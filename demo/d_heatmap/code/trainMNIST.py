import os
import torch
import argparse
from torch import nn, optim
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import shutil
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from utils.Logger import reproduc, MyLogger

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.AvgPool2d(kernel_size = 2,stride = 2),
            torch.nn.Tanh(),
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.Dropout2d(),
            torch.nn.AvgPool2d(kernel_size = 2,stride = 2),
            torch.nn.Tanh(),
            torch.nn.Flatten(),
            torch.nn.Linear(320, 50),
            torch.nn.Tanh(),
            torch.nn.Dropout2d(),
            torch.nn.Linear(50, 10)
        )

    def forward(self, x):
        y = self.net(x)
        return F.log_softmax(y)

def get_loss_function(loss_name):
    if loss_name == 'MSE':
        criterion = nn.MSELoss()
    elif loss_name == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_name == 'nll':
        criterion = F.nll_loss
    else:
        raise NotImplemented
    return criterion

def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    else:
        raise NotImplemented
    return optimizer

def train(opt, Log):
    # Dataset
    train_dataset = datasets.MNIST(root='demo/d_heatmap/data', train=True, transform=transforms.ToTensor(), download=opt.Dataset.download)
    train_loader  = DataLoader( train_dataset, batch_size=opt.Train.batch_size, shuffle=opt.Dataset.shuffle, num_workers=1)

    # Model, loss and optimizer
    model = MNISTNet()
    criterion = get_loss_function(opt.Train.loss)
    optimizer = get_optimizer(optimizer_name=opt.Train.optimizer, model=model, learning_rate=opt.Train.learning_rate)

    if torch.cuda.is_available():
        print("CUDA is enable!")
        model = model.cuda()
        model.train()

    # Training
    epoches_num = opt.Train.epoches_num
    check_point = opt.Train.check_point
    if check_point == -1:
        check_point = epoches_num
    loss_best = 10000000
    for epoch in range(epoches_num):
        print('*' * 40)
        train_loss = 0.0
        train_acc  = 0.0
        dataset_size = 0
        for batch_idx, (img, label) in enumerate(train_loader, 1 ):
            dataset_size += img.shape[0]

            if torch.cuda.is_available():
                img   = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img   = Variable(img)
                label = Variable(label)

            # Forward propagation
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            
            # Backward propagation
            loss.backward()
            optimizer.step()
            
            # Loss/accuracy
            train_loss += loss.item()
            predictions = torch.argmax(out, dim = 1)
            num_correct = predictions.eq(label).sum()
            train_acc  += num_correct.item()
        
        mean_loss = train_loss/dataset_size
        mean_acc = train_acc/dataset_size
        print('Finish  {}  Loss: {:.12f}, Acc: {:.12f}'.format(epoch+1 , mean_loss, mean_acc))
        Log.log_metrics({'Loss': mean_loss}, epoch+1)
        Log.log_metrics({'Acc': mean_acc}, epoch+1)

        if (epoch+1)%check_point == 0 and mean_loss<loss_best:
            loss_best = mean_loss
            save_path = os.path.join(Log.model_dir,'best.pt') 
            torch.save(model, save_path)
    save_path = os.path.join(Log.model_dir,'net.pt') 
    torch.save(model, save_path)

def main():
    opt = OmegaConf.load(args.p)
    Log = MyLogger(**opt['Log'])
    shutil.copy(args.p, Log.script_dir)
    shutil.copy(__file__, Log.script_dir)
    reproduc()
    train(opt, Log)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    parser = argparse.ArgumentParser(
        description='Parameter configuration of Taylor decomposition')
    parser.add_argument('-g', help='availabel gpu list', default='0',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-p', type=str, default='demo/d_heatmap/code/MNIST.yaml', help='yaml file path')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.g])

    main()
