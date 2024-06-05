from collections import Counter
import numpy as np

class TreeNode():
    def __init__(self, data, f_idx, f_val, information_gain) -> None:
        self.data = data
        self.f_idx = f_idx
        self.f_val = f_val
        #self.pred_prob = pred_prob
        self.information_gain = information_gain
        self.left = None
        self.right = None

        self.label = Counter(data[:, -1]).most_common()[0][0]

    def print(self, lables: list):
        if lables.any() == None:
            print(f'split on feature #{self.f_idx} with < {self.f_val}  label if ended here is: {self.label}',end='')
        else:
            print(f'split on {lables[self.f_idx]} with < {self.f_val}  label if ended here is: {self.label}',end='')

