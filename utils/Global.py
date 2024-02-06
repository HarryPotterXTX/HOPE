import math
import copy
import torch
import numpy as np

def create_coords(coords_shape, minimum, maximum):
    parameter = []
    for i in range(len(coords_shape)):
        parameter.append(torch.linspace(minimum[i],maximum[i],coords_shape[i]))
    coords = torch.stack(torch.meshgrid(parameter),axis=-1)
    coords = np.array(coords)
    return coords

# def taylor_output(grad_dict:dict, dx:np.array):
#     ''' 
#     Function: Calculate the value of Taylor polynomial at dx
#     Input: grad_dict: v = {'0':[y0], '1':[ay/a(x_1),...,ay/a(x_p)], ...}; dx: (batch, p)
#     Ouptut: y = \sum a^ky/a(x_i1)...a(x_ik)*(x_i1*...*x_ik)/k! = y0 + [v1/1!,v2/2!,...,vn//n!]*[x,x^{kron 2},...,x^{kron n}]^T
#     '''
#     n = max(grad_dict.keys())
#     coef = []
#     for k in range(1,n+1):
#         coef.append(np.array(grad_dict[k])/math.factorial(k))   # [v1/1!,v2/2!,...,vn//n!]
#     coef = np.concatenate(coef)
#     output = np.ones((dx.shape[0], 1))*grad_dict[0]
#     for i in range(dx.shape[0]):
#         xi, xkron_k, line = dx[i], dx[i], [dx[i]]
#         for k in range(2, n+1):
#             xkron_k = np.kron(xi, xkron_k)
#             line.append(xkron_k)    # [x,x^{kron 2},...,x^{kron n}]
#         output[i] = output[i] + np.dot(coef,np.concatenate(line))
#     return output

def taylor_output(grad_dict:dict, dx:np.array):
    ''' 
    Function: Calculate the value of Taylor polynomial at dx
    Input: grad_dict: v = {0:[y0], 1:[ay/a(x_1),...,ay/a(x_p)], ...}; dx: (batch, p)
    Ouptut: y = \sum a^ky/a(x_i1)...a(x_ik)*(x_i1*...*x_ik)/k!
    '''
    y = np.ones((dx.shape[0], 1))*grad_dict[0]              # y0
    assert len(dx.shape)==2, "The dimension of dx should be (batch, p)."
    dx = [dx[:,i:i+1] for i in range(dx.shape[1])]          # [x_1, x_2, ..., x_p]
    for k in range(1, max(grad_dict.keys())+1):
        idxs = create_coords([len(dx)]*k, [0]*k, [len(dx)-1]*k).astype(np.int16)
        idxs = idxs.reshape((-1, idxs.shape[-1]))           # eg. if k=3, p=2： [[000], [001], [010], ..., [111]]
        assert idxs.shape[0]==len(grad_dict[k]), "Error!"
        for i in range(idxs.shape[0]):
            idx = idxs[i]
            dy = 1
            for id in idx:
                dy = np.multiply(dy, dx[id])
            y = y + grad_dict[k][i]*dy/math.factorial(k)    # y = y + grad*(dx)^k/k!
    return y

def create_code(n:int, p:int):
    """
    Function: Create a list to encode the interaction terms. Code rule: eg. 4*x1^3*x2^1*x3^0 -> [4,3,1,0]
    Input: Expansion order n, input dimension p
    Output: Encoded list for [x, x^{kron 2}, ..., x^{kron n}].
    """
    code = []
    for k in range(1,n+1):
        idxs = create_coords([p]*k, [1]*k, [p]*k).astype(np.int16)
        idxs = idxs.reshape((-1, idxs.shape[-1]))   # eg. if k=2, p=3, idxs=[11,12,13,21,22,23,31,32,33]->[x1x1,x1x2,x1x3,...,x3x3]
        for i in range(idxs.shape[0]):
            line = [1] + [0]*p                      # eg. 1*x1^2*x2^1*x3^0 -> [1,2,1,0]
            for id in idxs[i]:
                line[id] += 1
            code.append(line)
    code = np.array(code)
    return code

def code_single_diff(code:np.array, var:int):
    """
    Function: Differential operations on code. eg. code=[4,3,1,0](4*x1^3*x2^1*x3^0), var=1(patial x1) -> [12,2,1,0](12*x1^2*x2^1*x3^0)
    Input: Code, partial order
    Output: The result of the partial differential
    """
    code = copy.deepcopy(code)
    code[:,0] = code[:,0]*code[:,var]
    code[:,var] = code[:,var] - 1
    code[code<0] = 0
    return code

def code_diff(code:np.array, vars:list):
    """
    Function: Differential operations on code. 
        eg. code=[4,3,1,0](4*x1^3*x2^1*x3^0), vars=[1,2,1](patial x1x2x1) 
            => [12,2,1,0]->[12,2,0,0]->[24,1,0,0](24*x1^1*x2^0*x3^0)
    Input: Code, partial order
    Output: The result of the partial differential
    """
    code = copy.deepcopy(code)
    for var in vars:
        code = code_single_diff(code, var)
    return code

def code2array(code:np.array, dx:np.array):
    '''
    Function: Decode to get the real data
    Input: code=[[a0,a1,a2,...,ap],...], dx=[dx1,dx2,...,dxp]
    Ouput: a0*dx1^a1*dx2^a2*...*dxp^ap
    '''
    assert len(dx.shape)==1 or (len(dx.shape)==2 and dx.shape[0]==1), "dx should be (1,p) or (p)"
    dx = dx[0] if len(dx.shape)==2 else dx
    result = code[:,0]
    for i in range(1,code.shape[1]):
        result = result*(dx[i-1]**code[:,i])
    return result

def dict_key_code(vars:list):   
    '''eg. [1,2,3]->'1_2_3' '''                 
    key = ''
    for var in vars:    
        key += str(var) + '_'
    return key[:-1]

def get_coef(grad_dict:dict, n:int):    
    '''return [v1/1!,v2/2!,...,vn/n!]'''
    coef = []
    for k in range(1,n+1):
        coef.append(np.array(grad_dict[k])/math.factorial(k))
    coef = np.concatenate(coef)
    return coef

def convert_derivatives(grad_dict:dict, x0:np.array, x1:np.array):
    '''
    Function: Convert the Taylor polynomial obtained at x0 to point x1
    Input: grad_dict: v = {0:[y0], 1:[ay/a(x_1),...,ay/a(x_p)], ...};
    Ouput:
    '''
    assert len(x0.shape)==2, "The dimension of dx should be (batch, p)."
    n, p = max(grad_dict.keys()), x0.shape[1]
    y1 = taylor_output(grad_dict=grad_dict, dx=x1-x0)   # polynomials output on point x1
    grad1_dict = {0:y1[0]}                              # polynomial's derivatives dict
    coef = get_coef(grad_dict, n)                       # The coefficients of Taylor polynomials on x0 [v1/1!,v2/2!,...,vn//n!]
    code_dict = {}                                      # code dict
    for k in range(1,n+1):
        idxs = create_coords([p]*k, [1]*k, [p]*k).astype(np.int16)
        idxs = idxs.reshape((-1, idxs.shape[-1]))       # eg. if k=2, p=3, idxs=[11,12,13,21,22,23,31,32,33]->[x1x1,x1x2,x1x3,...,x3x3]
        grad1_list = []
        for idx in idxs:
            key = dict_key_code(idx)
            sub_key = dict_key_code(idx[:-1])
            sub_code = create_code(n,p) if k==1 else code_dict[sub_key]
            sub_var = idx[-1]
            # Calculating based on the previously calculated code saves time and resources
            code_dict[key] = code_single_diff(sub_code, sub_var)
            # The k-order gradient on point x1
            xkron = code2array(code_dict[key], x1-x0)
            grad1_list.append(float((coef*xkron).sum()))
        grad1_dict[k] = grad1_list
    return grad1_dict