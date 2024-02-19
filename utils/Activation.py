import torch
from torch.distributions import normal

class SineAct():
    def __init__(self, **kwargs) -> None:
        pass
    def diff(self, input:torch.tensor, n:int):
        # cos -sin -cos sin 
        if n%4 ==1:
            return torch.cos(input)
        elif n%4 == 2:
            return -torch.sin(input)
        elif n%4 == 3:
            return -torch.cos(input)
        elif n%4 == 0:
            return torch.sin(input)

class ReLUAct():
    def __init__(self, **kwargs) -> None:
        pass
    def diff(self, output:torch.tensor, n:int):
        # n == 1 and output > 0: return 1; else: return 0
        gradient = torch.zeros_like(output)
        if n == 1:
            gradient[output>0] = 1
        return gradient

class LeakyReLUAct():
    def __init__(self, negative_slope, **kwargs) -> None:
        self.negative_slope = negative_slope
    def diff(self, input:torch.tensor, n:int):
        gradient = torch.zeros_like(input)
        if n == 1:
            gradient[input>0] = 1
            gradient[input<0] = self.negative_slope
            # gradient[input==0] = (1 + self.negative_slope)/2  # Loosely averaged
        return gradient

class NoneAct():
    def __init__(self, **kwargs) -> None:
        pass
    def diff(self, input:torch.tensor, n:int):
        if n == 1:
            gradient = torch.ones_like(input)
        else:
            gradient = torch.zeros_like(input)
        return gradient

class SigmoidAct():
    def __init__(self, order, **kwargs) -> None:
        self.order = order
        B = {}
        B[1] = {1:1}
        for k in range(2, self.order+2):
            B[k] = {1:1}                                # B[n+1][1] = B[n-1][1] = ... = B[1][1] = 1
            for i in range(2, k):
                B[k][i] = i*B[k-1][i]-(i-1)*B[k-1][i-1] # B_{k,i}=iB_{k-1,i}-(i-1)B_{k-1,i-1}, & i=2,\ldots,k-1
            B[k][k] = -(k-1)*B[k-1][k-1]                # B_{k,k}=-(k-1)B_{k-1,k-1}
        self.B = B
    def diff(self, output:torch.tensor, n:int):
        formula = self.B[n+1]
        gradient = torch.zeros_like(output)
        for key in formula.keys():
            power = int(key)
            gradient = gradient + int(formula[key])*torch.pow(output, power)
        return gradient

class TanhAct():
    def __init__(self, order, **kwargs) -> None:
        self.order = order
        C = {}
        C[1] = {1:1, 2:0}
        C[2] = {1:0, 2:1, 3:0}
        for k in range(3, self.order+3):
            C[k] = {}  
            C[k][1] = C[k-1][2]                                 # C_{k,1}=C_{k-1,2}, & k=2,\ldots,n+2
            for i in range(2, k):
                C[k][i] = i*C[k-1][i+1]-(i-2)*C[k-1][i-1]       # C_{k,i}=iC_{k-1,i+1}-(i-2)C_{k-1,i-1}, & i=2,\ldots,k-1
            C[k][k] = -(k-2)*C[k-1][k-1]                        # C_{k,k}=-(k-2)C_{k-1,k-1}
            C[k][k+1] = 0
        self.C = C
    def diff(self, output:torch.tensor, n:int):
        formula = self.C[n+2]
        gradient = torch.zeros_like(output)
        for key in formula.keys():
            power = int(key)-1
            gradient = gradient + int(formula[key])*torch.pow(output, power)
        return gradient

class ELUAct():
    '''f(x)=x if x>=0; a(e^x-1) else'''
    def __init__(self, alpha, **kwargs) -> None:
        self.alpha = alpha
    def diff(self, input:torch.tensor, n:int):
        gradient = torch.zeros_like(input)
        if n == 1:
            gradient[input>=0] = 1
        gradient[input<0] = self.alpha*torch.exp(input[input<0])
        return gradient
    
# class GELUAct():
#     '''Using Autograd to assist in calculating high-order derivatives'''
#     def __init__(self, order, approximate, **kwargs) -> None:
#         self.order = order
#         self.approximate = approximate
#         self.derivatives = {}
#     def diff(self, input:torch.tensor, n:int):
#         if self.order not in self.derivatives.keys():
#             act = torch.nn.GELU(approximate=self.approximate)
#             self.derivatives = {k:[] for k in range(0,self.order+1)}
#             pbar = tqdm(total=input.shape[-1]*self.order, desc='Calculating GELU\'s derivatives', leave=True, file=sys.stdout)
#             for i in range(input.shape[-1]):
#                 x = copy.deepcopy(input.data[...,i][0])
#                 x.requires_grad=True
#                 y = act(x)
#                 self.derivatives[0].append(y)
#                 for k in range(1, self.order+1):
#                     deri_last = self.derivatives[k-1][i]
#                     deri_now = torch.autograd.grad(deri_last, x, grad_outputs=torch.ones_like(deri_last), create_graph=True)[0]
#                     self.derivatives[k].append(deri_now)
#                     pbar.set_postfix_str("idx:{}/{}, order:{}/{}, ".format(i, input.shape[-1], k, self.order))
#                     pbar.update(1)
#         gradient = torch.tensor(self.derivatives[n])
#         if n == self.order: # clear
#             self.derivatives = {}
#         return gradient
    
class GELUAct():
    def __init__(self, order, **kwargs) -> None:
        self.order = order
        self.dist = normal.Normal(0, 1)
        D = {}
        D[1] = {1:2, 2:0, 3:-1}
        for k in range(2, self.order):
            D[k] = {}  
            D[k][1] = D[k-1][2]                                 # D_{k,1}=D_{k-1,2}, & k=2,\ldots,n-1
            D[k][k+1] = -D[k-1][k]                              # D_{k,k+1}=-D_{k-1,k}, & k=2,\ldots,n-1
            D[k][k+2] = -D[k-1][k+1]                            # D_{k,k+2}=-D_{k-1,k+1}, & k=2,\ldots,n-1
            for i in range(2, k+1):
                D[k][i] = i*D[k-1][i+1]-D[k-1][i-1]             # D_{k,i}=iD_{k-1,i+1}-D_{k-1,i-1}, & i=2,\ldots,k
        self.D = D
    def pdf(self, input:torch.tensor):
        return torch.exp(self.dist.log_prob(input))
    def diff(self, input:torch.tensor, n:int):
        if n == 1:
            gradient = self.dist.cdf(input) + input*self.pdf(input) 
        else:
            formula = self.D[n-1]
            gradient = torch.zeros_like(input)
            for key in formula.keys():
                power = int(key)-1
                gradient = gradient + int(formula[key])*torch.pow(input, power)
            gradient = gradient*self.pdf(input) 
        return gradient

ActDict = {'Sine':SineAct,'ReLU':ReLUAct,'LeakyReLU':LeakyReLUAct,'Sigmoid':SigmoidAct,'Tanh':TanhAct,'NoneAct':NoneAct}

def JudgeAct(ActClass, order):
    ActName = str(type(ActClass))
    if 'Sine' in ActName:
        return  ActDict['Sine']()
    elif 'LeakyReLU' in ActName:
        negative_slope = ActClass.negative_slope
        return  ActDict['LeakyReLU'](negative_slope)
    elif 'ReLU' in ActName:
        return  ActDict['ReLU']()
    elif 'Sigmoid' in ActName:
        return  ActDict['Sigmoid'](order)
    elif 'Tanh' in ActName:
        return  ActDict['Tanh'](order)
    elif 'None' in ActName:
        return ActDict['NoneAct']()
    else:
        raise NotImplemented