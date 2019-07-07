import os
# from path import Path
import numpy as np
from collections import defaultdict
import inspect

# tensor2DToCsv(tensor,path='/home/yj/Documents/Python/Github/seq2seq/data/gan.txt')
def tensor2DToCsv(tensor,path=None,token=',',write_name=True):

    def get_variable_name(variable):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return [var_name for var_name, var_val in callers_local_vars if var_val is variable]
    tensor = tensor.cpu()
    name = ''.join(get_variable_name(tensor))

    assert(path is not None)

    z = tensor.numpy().tolist()
    if len(np.shape(z)) == 2:
        with open(path,'a') as f:
            if write_name:
                f.write(name)
            else:
                f.write('\r')
            f.write('\r')
            for i in range(np.shape(z)[0]):
                for j in range(np.shape(z)[1]):
                    f.write(str(z[i][j]))
                    f.write(token)
                f.write('\r')
    elif len(np.shape(z)) == 1:
        with open(path,'a') as f:
            if write_name:
                f.write(name)
            else:
                f.write('\r')
            f.write('\r')
            for i in range(np.shape(z)[0]):
                f.write(str(z[i]))
                f.write(token)
            f.write('\r')
    else:
        raise "Not support 3D tensor."

# a = torch.tensor([[[1,2,3],[4,5,6]],[[2,2,2],[5,5,5]]])
# tensorToCsv(tensor,path='/home/yj/Documents/gan.txt')
def tensorToCsv(tensor,path=None,token=','):
    tensor = tensor.cpu().detach()
    z = tensor.numpy().tolist()
    if len(np.shape(z)) == 3:
        for i in range(np.shape(z)[0]):
            tensor2DToCsv(tensor[i],path=path,token=token,write_name=False)
    elif len(np.shape(z)) < 3:
        tensor2DToCsv(tensor,path=path,token=token)
    else:
        raise "Not support 4D tensor."

# import torch
# a = torch.tensor([[[1,2,3],[4,5,6]],[[2,2,2],[5,5,5]]])
# tensorToCsv(a,path='/home/yj/Documents/gan22.txt')

# from g_util import tensorToCsv
# tensorToCsv(tensor,path='/home/yj/Documents/gan.txt')