import torch
import torch.nn as nn 
from utils.Logger import reproduc
from utils.ChainMatrix import ChainTransMatrix, FormulaCal
from utils.Activation import ActDict

global class_dict
class_dict = {}

def load_class(name:str, **kwargs):
    if not name in class_dict.keys():
        if name == 'Chain':
            class_dict[name] = ChainTransMatrix(**kwargs) 
        elif name in ['Sine', 'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'NoneAct']:
            class_dict[name] = ActDict[name](**kwargs)
        else:
            raise NotImplemented
    return class_dict[name]

def hope_module(f, v:dict, order:int=1, device:str='cpu'):
    vs = []
    ######################### y = nn.Linear()(x): v[k].shape=(batch,num) #########################
    if type(f).__name__ == 'AddmmBackward0':    
        x, Wt = f._saved_mat1, f._saved_mat2
        assert len(x.shape)==2 and len(v[1].shape)==2, "x.shape should be (batch, num)"
        # 1. AccumulateGrad (b): v=v    
        vs.append(v)    
        # 2. **Backward0 (x): vx=M(W)vy (batch,num)->(batch,height,width)->(batch,num)
        vx = {k:torch.matmul(torch.pow(Wt, k), v[k].unsqueeze(-1)).squeeze(-1) for k in v.keys()}
        vs.append(vx)
        # 3. TBackward0 (W): v=\sum_batch M(x)v; v--(batch,num)->(batch,height,width); x--(batch,num)->(batch,width,height)
        vw = {k:torch.bmm(v[k].unsqueeze(-1), torch.pow(x.unsqueeze(1), k)) for k in v.keys()}
        vs.append(vw)        
                                              
    ######################### y = torch.bmm(x1, x2) #########################
    elif type(f).__name__ == 'BmmBackward0':
        x1, x2 = f._saved_self, f._saved_mat2
        assert len(x1.shape)==3 or len(x2.shape)==3, "The dimension of x1, x2 must be 3!" 
        # q=mn: a^ky/am^k=(a^ky/aq^k)*(n.T)^.k; a^ky/an^k=(m.T)^.k*(a^ky/aq^k)
        vx1 = {k:torch.bmm(v[k], torch.pow(x2.transpose(1, 2), k)) for k in v.keys()}
        vx2 = {k:torch.bmm(torch.pow(x1.transpose(1, 2), k), v[k]) for k in v.keys()}
        vs = [vx1, vx2]
    ######################### y = x.transpose(dim1, dim2) #########################
    elif type(f).__name__ == 'TransposeBackward0':      
        vx = {k:v[k].transpose(f._saved_dim0, f._saved_dim1) for k in v.keys()}
        vs.append(vx)
    ######################### y = nn.Flatten()(x) #########################
    elif type(f).__name__ == 'ReshapeAliasBackward0':   
        vx = {k:v[k].reshape(f._saved_self_sym_sizes) for k in v.keys()}
        vs.append(vx)
    ######################### y = torch.sin(x) #########################
    elif type(f).__name__ == 'SinBackward0':
        vx = {}
        x = f._saved_self
        chain_module, sine_module = load_class('Chain', n=order), load_class('Sine')
        Beta = {}
        for k in range(1, order+1):
            Beta[k] = sine_module.diff(x, k)
        for k in range(1, order+1):
            vx[k] = 0
            for s in range(1, k+1):
                vx[k] = vx[k] + torch.mul(FormulaCal(chain_module[k][s], Beta), v[s])
        vs.append(vx)
    ######################### y = x1/x2 #########################
    elif type(f).__name__ == 'DivBackward0':
        x1, x2 = f._saved_self, f._saved_other
        if x1 != None:
            raise NotImplemented
        vx = {k:v[k]/x2 for k in v.keys()}
        vs = [vx, None]
    elif type(f).__name__ == 'NoneType':                # y = c
        pass 
    # weight of linear layer: TBackward0 -> AccumulateGrad 
    elif type(f).__name__ == 'TBackward0':              
        vw = {k:torch.sum(v[k], 0) for k in v.keys()}   # v=\sum_batch v
        vs.append(vw)
    ######################### y = nn.Linear(a, b)(x): x.shape=(batch,1,num) #########################
    elif type(f).__name__ == 'ViewBackward0':
        # vx = {k:v[k].view(f._saved_self_sym_sizes) for k in v.keys()}
        # vs.append(vx)
        vs.append(v)
        raise NotImplemented
    else:
        raise  Exception(f"Module {type(f).__name__ } has not be developed!")
    return vs

def hopegrad(y:torch.tensor, order:int=10, device:str='cpu'):
    assert len(y.shape)==2, "The shape of y should be (batch, 1)!"
    f = y.grad_fn
    v = {i:torch.zeros((y.shape[0],1)).to(device) for i in range(1, order+1)}   # initial derivative vector
    v[1] = torch.ones((y.shape[0],1)).to(device)
    queue = [[f, v]]
    while queue != []:
        item = queue.pop()
        vs = hope_module(item[0], item[1], order, device)   # item[0](item[1])
        fs = [f[0] for f in item[0].next_functions]    
        # print([type(f).__name__ for f in fs], [v[1].shape if v!=None else None for v in vs])
        assert len(vs)==len(fs), "The number of vectors and functions should be the same!"
        for f, v in zip(fs, vs):    
            if type(f).__name__ == 'AccumulateGrad':        # leaf node
                v = {k:v[k].detach() for k in v.keys()}
                f.variable.hope_grad = {k:f.variable.hope_grad[k]+v[k] for k in v.keys()} if hasattr(f.variable, 'hope_grad') else v
            elif f!=None:
                queue.append([f, v])