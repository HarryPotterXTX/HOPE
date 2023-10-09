import copy
import torch
from typing import Dict, List

def SingleStrFormula(single_formula:List):
    """
    Function: Str formula. bj is g(x)'s j-order derivative
    Input: Coefficient, order and power of bj.
    Output: Formula. eg. [3,1,2,3,4] -> 3b1^2b3^4
    """
    formula = ''
    if single_formula[0] > 1:
        formula += f'{single_formula[0]}'
    for i in range(1,len(single_formula),2):
        formula += f'b{single_formula[i]}'
        if single_formula[i+1] > 1:
            formula += f'^{single_formula[i+1]}'
    return formula

def StrFormula(formulas:List):
    """
    eg. [[1, 2, 6], [5, 1, 1, 2, 4, 3, 1]] -> b2^6 + 5b1b2^4b3
    """
    formula = ''
    for i in range(len(formulas)):
        formula += SingleStrFormula(formulas[i])
        if i < len(formulas)-1:
            formula += ' + '
    return formula


def MultiFormula(order:int, power:int, formula:List):
    """
    Function: Formula multiplied by bj^k
    eg. (2, 3, [[1,1,3,3,4], [3,2,2,5,3], [1,1,3]]) (b2^3*(b1^3b3^4 + 3b2^2b5^3 + b1^3))
        -> [[1, 1, 3, 2, 3, 3, 4], [3, 2, 5, 5, 3], [1, 1, 3, 2, 3]] (b1^3b2^3b3^4 + 3b2^5b5^3 + b1^3b2^3)
    """
    formula = copy.deepcopy(formula)
    result = []
    for single_formula in formula:
        for i in range(1,len(single_formula),2):
            if single_formula[i] == order:
                single_formula[i+1] += power
                break
            elif single_formula[i] > order:
                single_formula.insert(i,power)
                single_formula.insert(i,order)
                break
        if single_formula[-2] < order:
            single_formula.append(order)
            single_formula.append(power)
        result.append(single_formula)
    return result

def AddFormula(formula1:List, formula2:List):
    """
    Function: Addition formula
    eg. [[2,2,2,5,3]], [[3,2,2,5,3], [1,1,3]]
        -> [[5,2,2,5,3], [1,1,3]]
    """
    formula = copy.deepcopy(formula1 + formula2)
    result = []
    Flag = []
    Index = []
    for i in range(len(formula)):
        single_formula = formula[i]
        order_power = single_formula[1:]
        if order_power not in Flag:
            Flag.append(order_power)
            Index.append(len(result))
            result.append(single_formula)
        else:
            result[Index[Flag.index(order_power)]][0] += single_formula[0]
    return result

def SingleDiff(single_formula:List):
    """
    Function: Single partial differential. bj is g(x)'s j-order derivative
    Input: Coefficient, order and power of bj. eg. [3,1,2,3,4] (3b1^2b3^4), [4,2,1,1,4] (4b2^1b1^4)
    Output: The result of the partial differential. eg. [3,1,2,3,4] (3b1^2b3^4) -> [[6,1,1,2,1,3,4], [12,1,2,3,3,4,1]] (6b1b2b3^4+12b1^2b3^3b4)
    """
    result = []
    for i in range(1,len(single_formula),2):
        tem = copy.deepcopy(single_formula)
        tem[0] *= single_formula[i+1]   # coefficient
        tem += [single_formula[i]+1, 1] # order
        tem[i+1] -= 1                   # power
        # clear power 0
        if tem[i+1] == 0:
            tem.pop(i)
            tem.pop(i)
        # power merge (bi^mbi^n -> bi^(m+n)) and order sort (b3b1b6b8 -> b1b3b6b8)
        orders = []
        powers = []
        for j in range(1,len(tem),2):
            order = tem[j]
            power = tem[j+1]
            if order not in orders:
                orders.append(order)
                powers.append(power)
            else:
                powers[orders.index(order)] += power
        rows = []
        for i in range(len(orders)):
            row = {'order':orders[i],'power':powers[i]}
            rows.append(row)
        rows = sorted(rows, key=lambda r:r['order'], reverse=False)
        tem = [tem[0]]
        for i in range(len(rows)):
            tem.append(rows[i]['order'])
            tem.append(rows[i]['power'])
        result.append(tem)
    return result

def Diff(formula:List):
    """
    Function: Partial differential. bj is g(x)'s j-order derivative
    eg. [[1,1,1], [3,1,2,3,4], [1,1,1,2,2,3,3]] (b1 + 3b1^2b3^4 + b1b2^2b3^3) 
        -> [[1, 2, 1], [8, 1, 1, 2, 1, 3, 4], [12, 1, 2, 3, 3, 4, 1], [1, 2, 3, 3, 3], [3, 1, 1, 2, 2, 3, 2, 4, 1]] (b2 + 8b1b2b3^4 + 12b1^2b3^3b4 + b2^3b3^3 + 3b1b2^2b3^2b4)
    """
    result = []
    Flag = []
    Index = []
    for i in range(len(formula)):
        single_formula = formula[i]
        tem_formula = SingleDiff(single_formula)
        for tem_single in tem_formula:
            order_power = tem_single[1:]
            if order_power not in Flag:
                Flag.append(order_power)
                Index.append(len(result))
                result.append(tem_single)
            else:
                result[Index[Flag.index(order_power)]][0] += tem_single[0]
    return result
            
def ChainTransMatrix(order:int):
    """
    Function: Find the n-order chain transformation matrix expression
    """
    BnCal = {}
    for i in range(1,order+1):
        BnCal[i] = {}
    BnCal[1][1] = [[1,1,1]]
    for i in range(2,order+1):
        for j in range(1,i+1):
            # Bij = aB(i-1,j)/ax + ag/ax*B(i-1,j-1)
            if j == 1:
                BnCal[i][j] = [[1,i,1]]
            elif j == i:
                BnCal[i][j] = [[1,1,i]]
            else:
                BnCal[i][j] = AddFormula(Diff(BnCal[i-1][j]), MultiFormula(1,1,BnCal[i-1][j-1]))
    return BnCal

def SingleFormulaCal(formula:list, derivatives:Dict):
    """
    eg. [2, 4, 5], [B1, B2, ..., Bn] -> 2*B4.^5 (dot multiplication)
    """
    result = formula[0]
    for i in range(1,len(formula),2):
        order = formula[i]
        power = formula[i+1]
        result = torch.mul(result, torch.pow(derivatives[order],power))
    return result

def FormulaCal(formula:list, derivatives:Dict):
    """
    eg. [[1, 1, 1], [2, 4, 5]], [B1, B2, ..., Bn] -> 1*B1.^1 + 2*B4.^5 (dot multiplication)
    """
    result = 0
    for single_formula in formula:
        result = result + SingleFormulaCal(single_formula, derivatives)
    return result

def ChainCal(BnCal:Dict, derivatives:Dict):
    n = len(BnCal.keys())
    shape = derivatives[1].shape
    result = torch.zeros((shape[0], shape[1]*n, shape[2]*n))
    for key in BnCal.keys():
        for sub_key in BnCal[key].keys():
            formula = BnCal[key][sub_key]
            x1, x2, y1, y2 = shape[-2]*(int(key)-1), shape[-2]*int(key), shape[-1]*(int(sub_key)-1), shape[-1]*int(sub_key)
            result[:,x1:x2,y1:y2] = FormulaCal(formula, derivatives)
    return result