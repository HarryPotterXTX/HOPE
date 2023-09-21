import torch
import torch.nn as nn 
import torch.nn.functional as F
from utils.Logger import reproduc
from utils.ChainMatrix import ChainTransMatrix, FormulaCal
from utils.Activation import SineAct, SigmoidAct, TanhAct

global common_modules
common_modules = {}
ActDict = {'SinBackward0':SineAct, 'SigmoidBackward0':SigmoidAct, 'TanhBackward0':TanhAct}

def load_class(name:str, **kwargs):
    # Save commonly used components to avoid time waste of initialization
    if not name in common_modules.keys():
        if name == 'Chain':
            common_modules[name] = ChainTransMatrix(**kwargs) 
        else:
            common_modules[name] = ActDict[name](**kwargs)
    return common_modules[name]

def hope_module(f, v:dict, order:int=1, device:str='cpu', mixed:bool=False):
    vs = []
    module = type(f).__name__
    ######################### y = nn.Linear()(x): v[k].shape=(batch,num) #########################
    if module == 'AddmmBackward0':    
        x, Wt = f._saved_mat1, f._saved_mat2
        assert len(x.shape)==2 and len(v[1].shape)==2, "x.shape should be (batch, num)"
        # 1. AccumulateGrad (b): v=v    
        vs.append(v)    
        # 2. **Backward0 (x): vx=M(W)vy (batch,num)->(batch,height,width)->(batch,num)
        if mixed == True and Wt.shape[0] > 1 and type(f.next_functions[1][0]).__name__ == 'AccumulateGrad':   
            vx = {}                             # (1) mixed partial derivatives
            dx, dy = Wt.shape
            Wk = torch.ones(1, dy)
            for k in range(1, order+1):
                augW1 = torch.kron(Wt.contiguous(), torch.ones(dx**(k-1), 1))  
                augW2 = torch.kron(torch.ones(dx, 1), Wk) 
                Wk = torch.mul(augW1, augW2)
                vx[k] = torch.matmul(Wk, v[k].unsqueeze(-1))
        else:                                   # (2) unmixed partial derivatives
            vx = {k:torch.matmul(torch.pow(Wt, k), v[k].unsqueeze(-1)).squeeze(-1) for k in v.keys()}
        vs.append(vx)
        # 3. TBackward0 (W): v=\sum_batch M(x)v; v--(batch,num)->(batch,height,width); x--(batch,num)->(batch,width,height)
        vw = {k:torch.bmm(v[k].unsqueeze(-1), torch.pow(x.unsqueeze(1), k)) for k in v.keys()}
        vs.append(vw)  
    ######################### y = x1 + x2 #########################     
    elif module == 'AddBackward0':
        vs = [v, v]                                
    ######################### y = torch.bmm(x1, x2) #########################
    elif module == 'BmmBackward0':
        x1, x2 = f._saved_self, f._saved_mat2
        assert len(x1.shape)==3 and len(x2.shape)==3, "The dimension of x1, x2 must be 3!" 
        # q=mn: a^ky/am^k=(a^ky/aq^k)*(n.T)^.k; a^ky/an^k=(m.T)^.k*(a^ky/aq^k)
        vx1 = {k:torch.bmm(v[k], torch.pow(x2.transpose(1, 2), k)) for k in v.keys()}
        vx2 = {k:torch.bmm(torch.pow(x1.transpose(1, 2), k), v[k]) for k in v.keys()}
        vs = [vx1, vx2]
    ######################### y = x.transpose(dim1, dim2) #########################
    elif module == 'TransposeBackward0':      
        vx = {k:v[k].transpose(f._saved_dim0, f._saved_dim1) for k in v.keys()}
        vs.append(vx)
    ######################### y = nn.Flatten()(x) #########################
    elif module == 'ReshapeAliasBackward0':   
        vx = {k:v[k].reshape(f._saved_self_sym_sizes) for k in v.keys()}
        vs.append(vx)
    ######################### y = act(x) #########################
    elif module in ['SinBackward0', 'SigmoidBackward0', 'TanhBackward0']:
        vx = {}
        chain_module, act_module = load_class('Chain', order=order), load_class(module, order=order)
        if module == 'SinBackward0':        # the input of Sine
            data = f._saved_self
        elif module == 'SigmoidBackward0':  # the output of Sigmoid
            data = f._saved_result
        elif module == 'TanhBackward0':     # the output of Tanh
            data = f._saved_result
        Beta = {}
        for k in range(1, order+1):
            Beta[k] = act_module.diff(data, k)
        for k in range(1, order+1):
            vx[k] = 0
            for s in range(1, k+1):
                vx[k] = vx[k] + torch.mul(FormulaCal(chain_module[k][s], Beta), v[s])
        vs.append(vx)
    ######################### y = x1/x2 #########################
    elif module == 'DivBackward0':
        x1, x2 = f._saved_self, f._saved_other
        if x1 != None:
            raise NotImplemented
        vx = {k:v[k]/(x2**k) for k in v.keys()}
        vs = [vx, None]
    elif module == 'NoneType':                # y = c
        pass 
    ######################### weight of linear layer: TBackward0 -> AccumulateGrad ######################### 
    elif module == 'TBackward0':              
        vw = {k:torch.sum(v[k], 0) for k in v.keys()}   # v=\sum_batch v
        vs.append(vw)
    ######################### y = x.view(ouptut_size) #########################
    elif module == 'ViewBackward0':
        vx = {k:v[k].view(f._saved_self_sym_sizes) for k in v.keys()}
        vs.append(vx)
    ######################### y = nn.Dropout(attn_dropout)(x): y=torch.mul(x, y.grad_fn._saved_other) #########################
    elif module == 'MulBackward0':
        x1, x2 = f._saved_self, f._saved_other
        vx1 = {k:torch.mul(torch.pow(x2, k), v[k]) for k in v.keys()} if x2 != None else None
        vx2 = {k:torch.mul(torch.pow(x1, k), v[k]) for k in v.keys()} if x1 != None else None
        vs = [vx1, vx2]
    ######################### y = nn.nn.AvgPool2d()(x): z = conv(x, W) + b: a^ky/ax^k = conv(a^ky/az^k, rotate(W^(ok))) #########################
    elif module == 'ConvolutionBackward0':
        stride, padding, dilation = f._saved_stride, f._saved_padding, f._saved_dilation
        x, W = f._saved_input, f._saved_weight  # x: (batch channel height width); W: (outdim channel height width)
        # 1. **Backward0 (x)
        output_padding = ((x.shape[-2]-W.shape[-2]+2*padding[0])%stride[0], (x.shape[-1]-W.shape[-1]+2*padding[1])%stride[1])
        vx = {k:F.conv_transpose2d(input=v[k], weight=torch.pow(W, k), stride=stride, padding=padding, output_padding=output_padding, dilation=dilation) for k in v.keys()}
        # 2. AccumulateGrad (W) # TODO
        vw = None
        # 3. AccumulateGrad (b) # TODO
        vb = None
        vs = [vx, vw, vb]
    ######################### y = nn.AvgPool2d()(x) #########################
    elif module == 'AvgPool2DBackward0':
        kernel_size, stride, padding = f._saved_kernel_size, f._saved_stride, f._saved_padding
        x = f._saved_self   # x: (batch channel height width); W: (outdim=channel channel height width)
        W = torch.zeros(x.shape[-3], x.shape[-3], kernel_size[0], kernel_size[1])
        for i in range(W.shape[0]):
            W[i][i] = torch.ones(kernel_size[0], kernel_size[1])/(kernel_size[0]*kernel_size[1])
        output_padding = ((x.shape[-2]-W.shape[-2]+2*padding[0])%stride[0], (x.shape[-1]-W.shape[-1]+2*padding[1])%stride[1])
        vx = {k:F.conv_transpose2d(input=v[k], weight=torch.pow(W, k), stride=stride, padding=padding, output_padding=output_padding) for k in v.keys()}
        vs = [vx]
    ######################### y = nn.Softmax(dim=dim)(x): y = torch.exp(x)/torch.exp(x).sum(dim=set_dim).unsqueeze(set_dim) #########################
    elif module == 'SoftmaxBackward0':
        dim, y = f._saved_dim, f._saved_result
        raise NotImplemented
    else:
        raise Exception(f"Module {module} has not be developed!")
    return vs

def hopegrad(y:torch.tensor, order:int=10, device:str='cpu', mixed:bool=False):
    assert len(y.shape)==2, "The shape of y should be (batch, 1)!"
    f = y.grad_fn
    v = {i:torch.zeros((y.shape[0],1)).to(device) for i in range(1, order+1)}   # initial derivative vector
    v[1] = torch.ones((y.shape[0],1)).to(device)
    queue = [[f, v]]
    while queue != []:
        item = queue.pop()
        vs = hope_module(f=item[0], v=item[1], order=order, device=device, mixed=mixed)
        fs = [f[0] for f in item[0].next_functions]    
        print([type(f).__name__ for f in fs], [v[1].shape if v!=None else None for v in vs])
        assert len(vs)==len(fs), "The number of vectors and functions should be the same!"
        for f, v in zip(fs, vs):    
            if type(f).__name__ == 'AccumulateGrad':        # leaf node
                if v != None:
                    v = {k:v[k].detach() for k in v.keys()}
                f.variable.hope_grad = {k:f.variable.hope_grad[k]+v[k] for k in v.keys()} if hasattr(f.variable, 'hope_grad') else v
            elif f!=None:
                queue.append([f, v])