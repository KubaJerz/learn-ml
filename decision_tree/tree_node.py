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