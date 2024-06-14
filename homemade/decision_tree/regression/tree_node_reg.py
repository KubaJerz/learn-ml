from collections import Counter
import numpy as np

class TreeNode():
    def __init__(self, data, f_idx, f_val, information_gain) -> None:
        self.data = data
        self.f_idx = f_idx
        self.f_val = f_val
        self.information_gain = information_gain
        self.left = None
        self.right = None

        self.val = np.mean(data[:, -1])

    def print(self, val: list):
        if val == None:
            print(f'split on feature #{self.f_idx} with < {self.f_val}  label if ended here is: {self.val}',end='')
        elif val.any() == None:
            print(f'split on feature #{self.f_idx} with < {self.f_val}  label if ended here is: {self.val}',end='')
        elif np.isscalar(val):
            print(f'split on {val} with < {self.f_val}  label if ended here is: {self.val}',end='')
        else:
            print(f'split on {val[self.f_idx]} with < {self.f_val}  label if ended here is: {self.val}',end='')

