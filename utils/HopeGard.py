import torch
import torch.nn as nn 
import math
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

def cal_act_vx(Beta, vz):
    ''' x -> act function -> z -> other modules -> y
            Beta = {a^kz/ax^k}; vz = {a^ky/az^k}; vx = M(Beta)*vz = {a^ky/ax^k}
    '''
    chain_module = load_class('Chain', order=len(Beta.keys()))
    vx, order = {}, len(Beta.keys())
    for k in range(1, order+1):
        vx[k] = 0
        for s in range(1, k+1):
            vx[k] = vx[k] + torch.mul(FormulaCal(chain_module[k][s], Beta), vz[s])
    return vx

def hope_module(f, vz:dict, order:int=1, device:str='cpu', mixed:bool=False):
    ''' x -> module -> z -> other modules -> y: 
            we have vz[k] = [a^ky/az^k], and we want to calculate vx[k] = [a^ky/ax^k]
    '''
    vs = []
    module = type(f).__name__
    ######################### z = nn.Linear()(x): v[k].shape=(batch,num) #########################
    if module == 'AddmmBackward0':    
        x, Wt = f._saved_mat1, f._saved_mat2
        assert len(x.shape)==2 and len(vz[1].shape)==2, "x.shape should be (batch, num)"
        # 1. AccumulateGrad (b): vx = vz    
        # 2. **Backward0 (x): vx=M(W)vy (batch,num)->(batch,height,width)->(batch,num)
        if mixed == True and Wt.shape[0] > 1 and type(f.next_functions[1][0]).__name__ == 'AccumulateGrad':   
            vx = {}                             # (1) mixed partial derivatives
            dx, dz = Wt.shape
            Wk = torch.ones(1, dz)
            for k in range(1, order+1):
                augW1 = torch.kron(Wt.contiguous(), torch.ones(dx**(k-1), 1))  
                augW2 = torch.kron(torch.ones(dx, 1), Wk) 
                Wk = torch.mul(augW1, augW2)
                vx[k] = torch.matmul(Wk, vz[k].unsqueeze(-1))
        else:                                   # (2) unmixed partial derivatives
            vx = {k:torch.matmul(torch.pow(Wt, k), vz[k].unsqueeze(-1)).squeeze(-1) for k in vz.keys()}
        # 3. TBackward0 (W): v=\sum_batch M(x)v; v--(batch,num)->(batch,height,width); x--(batch,num)->(batch,width,height)
        vw = {k:torch.bmm(vz[k].unsqueeze(-1), torch.pow(x.unsqueeze(1), k)) for k in vz.keys()}
        vs = [vz, vx, vw]  
    ######################### z = x1 + x2 #########################     
    elif module == 'AddBackward0':
        vs = [vz, vz]                                
    ######################### z = torch.bmm(x1, x2): x1 and x2 must be independent of each other #########################
    elif module == 'BmmBackward0':
        x1, x2 = f._saved_self, f._saved_mat2
        assert len(x1.shape)==3 and len(x2.shape)==3, "The dimension of x1, x2 must be 3!" 
        # q=mn: a^ky/am^k=(a^ky/aq^k)*(n.T)^.k; a^ky/an^k=(m.T)^.k*(a^ky/aq^k)
        vx1 = {k:torch.bmm(vz[k], torch.pow(x2.transpose(1, 2), k)) for k in vz.keys()}
        vx2 = {k:torch.bmm(torch.pow(x1.transpose(1, 2), k), vz[k]) for k in vz.keys()}
        vs = [vx1, vx2]
    ######################### z = x.transpose(dim1, dim2) #########################
    elif module == 'TransposeBackward0':      
        vx = {k:vz[k].transpose(f._saved_dim0, f._saved_dim1) for k in vz.keys()}
        vs.append(vx)
    ######################### z = nn.Flatten()(x) #########################
    elif module == 'ReshapeAliasBackward0':   
        vx = {k:vz[k].reshape(f._saved_self_sym_sizes) for k in vz.keys()}
        vs.append(vx)
    ######################### z = act(x) #########################
    elif module in ['SinBackward0', 'SigmoidBackward0', 'TanhBackward0']:
        Beta = {}
        act_module = load_class(module, order=order)
        if module == 'SinBackward0':        # the input of Sine
            data = f._saved_self
        elif module == 'SigmoidBackward0':  # the output of Sigmoid
            data = f._saved_result
        elif module == 'TanhBackward0':     # the output of Tanh
            data = f._saved_result
        for k in range(1, order+1):
            Beta[k] = act_module.diff(data, k)
        vs.append(cal_act_vx(Beta, vz))
    ######################### z = c/x = c*(1/x): zi = c/xi (similar to a nolinear activation function) #########################
    elif module == 'ReciprocalBackward0':
        Beta = {}
        z = f._saved_result
        for k in range(1, order+1):
            Beta[k] = -(-z)**(k+1)*math.factorial(k)
        vs.append(cal_act_vx(Beta, vz))
    ######################### z = x**p (similar to a nolinear activation function) #########################
    elif module == 'PowBackward0':
        Beta = {}
        p, fact, x = f._saved_exponent, f._saved_exponent, f._saved_self
        for k in range(1, order+1):
            Beta[k] = fact*x**(p-k)
            fact = fact*(p-k)
        vs.append(cal_act_vx(Beta, vz))
    ######################### z = x1/x2 #########################
    elif module == 'DivBackward0':
        x1, x2 = f._saved_self, f._saved_other
        if x1 != None:
            raise NotImplemented
        vx = {k:vz[k]/(x2**k) for k in vz.keys()}
        vs = [vx, None] 
    ######################### weight of linear layer: TBackward0 -> AccumulateGrad ######################### 
    elif module == 'TBackward0':              
        vw = {k:torch.sum(vz[k], 0) for k in vz.keys()}   # v=\sum_batch v
        vs.append(vw)
    ######################### z = x.view(ouptut_size) #########################
    elif module == 'ViewBackward0':
        vx = {k:vz[k].view(f._saved_self_sym_sizes) for k in vz.keys()}
        vs.append(vx)
    ######################### z = nn.Dropout(attn_dropout)(x): z=torch.mul(x, y.grad_fn._saved_other) #########################
    elif module == 'MulBackward0':
        x1, x2 = f._saved_self, f._saved_other
        vx1 = {k:torch.mul(torch.pow(x2, k), vz[k]) for k in vz.keys()} if x2 != None else None
        vx2 = {k:torch.mul(torch.pow(x1, k), vz[k]) for k in vz.keys()} if x1 != None else None
        vs = [vx1, vx2]
    ######################### z = conv(x, W) + b: a^ky/ax^k = conv(a^ky/az^k, rotate(W^(ok))) #########################
    elif module == 'ConvolutionBackward0':
        stride, padding, dilation = f._saved_stride, f._saved_padding, f._saved_dilation
        x, W = f._saved_input, f._saved_weight  # x: (batch channel height width); W: (outdim channel height width)
        # 1. **Backward0 (x)
        output_padding = ((x.shape[-2]-W.shape[-2]+2*padding[0])%stride[0], (x.shape[-1]-W.shape[-1]+2*padding[1])%stride[1])
        vx = {k:F.conv_transpose2d(input=vz[k], weight=torch.pow(W, k), stride=stride, padding=padding, output_padding=output_padding, dilation=dilation) for k in vz.keys()}
        # 2. AccumulateGrad (W) # TODO
        vw = None
        # 3. AccumulateGrad (b) # TODO
        vb = None
        vs = [vx, vw, vb]
    ######################### z = nn.AvgPool2d()(x) #########################
    elif module == 'AvgPool2DBackward0':
        kernel_size, stride, padding = f._saved_kernel_size, f._saved_stride, f._saved_padding
        x = f._saved_self   # x: (batch channel height width); W: (outdim=channel channel height width)
        W = torch.zeros(x.shape[-3], x.shape[-3], kernel_size[0], kernel_size[1])
        for i in range(W.shape[0]):
            W[i][i] = torch.ones(kernel_size[0], kernel_size[1])/(kernel_size[0]*kernel_size[1])
        output_padding = ((x.shape[-2]-W.shape[-2]+2*padding[0])%stride[0], (x.shape[-1]-W.shape[-1]+2*padding[1])%stride[1])
        vx = {k:F.conv_transpose2d(input=vz[k], weight=torch.pow(W, k), stride=stride, padding=padding, output_padding=output_padding) for k in vz.keys()}
        vs = [vx]
    ######################### z = c #########################
    elif module == 'NoneType':
        pass
    ######################### z = nn.Softmax(dim=dim)(x): z = torch.exp(x)/torch.exp(x).sum(dim=set_dim).unsqueeze(set_dim) #########################
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
        vs = hope_module(f=item[0], vz=item[1], order=order, device=device, mixed=mixed)
        fs = [f[0] for f in item[0].next_functions]    
        # print([type(f).__name__ for f in fs], [v[1].shape if v!=None else None for v in vs])
        assert len(vs)==len(fs), "The number of vectors and functions should be the same!"
        for f, v in zip(fs, vs):    
            if type(f).__name__ == 'AccumulateGrad':        # leaf node
                if v != None:
                    v = {k:v[k].detach() for k in v.keys()}
                f.variable.hope_grad = {k:f.variable.hope_grad[k]+v[k] for k in v.keys()} if hasattr(f.variable, 'hope_grad') else v
            elif f!=None:
                queue.append([f, v])