from collections import Counter
import math
import numpy as np
from tree_node_reg import TreeNode
import pandas as pd

class DecisionTree():

    def __init__(self, max_depth = 5, min_samples = 0, min_var_decrease = 0.0, criterion = 'mse', num_rand_features = -1) -> None:
        self.max_depth = max_depth #mac depth the tree can be
        self.min_samples = min_samples #min samples allowed in leaf for us to continues to split on it 
        self.min_var_decrease = min_var_decrease #min reduction invariance needed to continue to build the tree
        self.val = None
        self.criterion = criterion
        self.num_rand_features = num_rand_features #the number of random features to concider when making split. so if n_r_f = 5 it will only concider 5 out of all time random features


    def __mae(self, all_vals: list) -> float:
        mean = np.mean(all_vals)
        return sum(all_vals - mean) / len(all_vals)
    
    def __mse(self, all_vals: list) -> float:
        mean = np.mean(all_vals)
        return sum((all_vals - mean) ** 2) / len(all_vals)
    

    def __sets_mae(self, splits: list) -> float:
        return sum(self.__mae(split) for split in splits)

    def __sets_mse(self, splits: list) -> float:
        return sum(self.__mse(split) for split in splits)

    def __split(self, data, f_index, threashold):
        bool_mask = data[:, f_index] < threashold 
        left = data[bool_mask]
        right = data[~bool_mask]
        return left, right

    '''
    We will take the greedy approach so,
    Given some data the best split will be on the feature that returns two sets with the lowest entrpoy.
    '''
    def __find_best_split(self, data: np.array) -> tuple:
        def evaluate_split(data, features_idx_list):
            criterion = self.criterion
            data_left_best = np.empty(0)
            data_right_best = np.empty(0)
            b_s = float('inf')
            
            for f_idx in features_idx_list:
                f_values = np.unique(data[:, f_idx])  # Ensure unique values
                
                for possible_threshold in f_values:
                    data_left, data_right = self.__split(data, f_idx, possible_threshold)

                    if len(data_left) > 0 and len(data_right) > 0:
                        if criterion == 'mae':
                            split_gain = self.__sets_mae([data_left[:, -1], data_right[:, -1]])
                        else:
                            split_gain = self.__sets_mse([data_left[:, -1], data_right[:, -1]])

                        if split_gain < b_s:
                            b_s = split_gain
                            data_right_best = data_right
                            data_left_best = data_left
                            best_f_idx = f_idx
                            best_f_val = possible_threshold

            if data_left_best.size == 0 or data_right_best.size == 0: #checks for odd bug where if the new data was a duplicate there the split was [], [dat1, data2] and then the if statemtn above was never entered causing issues
                return data, data, -1, None, 0
            return data_left_best, data_right_best, best_f_idx, best_f_val, b_s

        if self.num_rand_features == -1 or self.num_rand_features > len(data[0]) - 1:
            features_idx_list = range(data.shape[1] - 1)
        else:
            features_idx_list = np.random.randint(0, data.shape[1] - 1, size=self.num_rand_features)

        return evaluate_split(data, features_idx_list)

    

    '''
    builds the tree by recursivly doing the best (greedy) split until one of the stopping conditions have been met 
    '''
    def __build_tree(self, data: np.array, curr_depth: int) -> TreeNode:
        if curr_depth >= self.max_depth:
            return None

        s_data_left, s_data_right, s_f_idx, s_f_val, _ = self.__find_best_split(data)

        # calc the variance of the curr node
        current_variance = np.var(data[:, -1])

        # calc the weighted variance of the children
        n_left, n_right = len(s_data_left), len(s_data_right)
        n_total = n_left + n_right
        if n_total > 0:
            weighted_child_variance = (
                (n_left / n_total) * np.var(s_data_left[:, -1]) +
                (n_right / n_total) * np.var(s_data_right[:, -1])
            )
        else:
            weighted_child_variance = 0

        # calc variance reduction 
        variance_reduction = current_variance - weighted_child_variance

        node = TreeNode(data, s_f_idx, s_f_val, variance_reduction)

        if variance_reduction <= 0 or variance_reduction < self.min_var_decrease:
            return node

        if self.min_samples >= s_data_left.shape[0] and self.min_samples >= s_data_right.shape[0]:
            return node

        curr_depth += 1
        node.left = self.__build_tree(s_data_left, curr_depth)
        node.right = self.__build_tree(s_data_right, curr_depth)

        return node
    
    '''
    trains the tree
    '''
    def fit(self, X_train: np.array, y_train: np.array):
        train_data = np.concatenate((np.reshape(X_train,(len(X_train),-1)), np.reshape(y_train,(-1,1))),axis = 1)

        self.tree = self.__build_tree(data=train_data, curr_depth=0)

    def predict_one_sample(self, x: np.array) -> int:
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        pred = self.predict_helper(x, self.tree)
        return pred

    '''
    predicts for set samples
    '''
    def predict(self, X: np.array) -> np.array:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        preds = [self.predict_helper(x, self.tree) for x in X]
        return preds


    def predict_helper(self, x: np.array, tree: TreeNode) -> np.array:
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy() 
        if tree.left == None and tree.right == None:
            return tree.val
        split_index = tree.f_idx

        #check if its a scaler
        if np.isscalar(x):
            if x < tree.f_val:
                return self.predict_helper(x, tree.left)
            else:
                return self.predict_helper(x, tree.right)
        else:
            if x[split_index] < tree.f_val:
                return self.predict_helper(x, tree.left)
            else:
                return self.predict_helper(x, tree.right)
        
        
    def set_labels(self, val):
        self.val = val
        
    def print(self):
        level = 0
        print('    '*3,'level: ',level,end='     ')
        self.tree.print(self.val)
        print('\n')
        self.__print_helper(self.tree, level)


    def __print_helper(self, node: TreeNode, level: int):
        if node.left == None or node.right == None:
            return
        else:
            level += 1
            print('Level: ',level,'  LEFT: ',end='')
            node.left.print(self.val)
            print('\n')
            self.__print_helper(node.left, level)
            print('level: ',level,'  RIGHT:  ',end='')
            node.right.print(self.val)
            print('\n')
            self.__print_helper(node.right, level)

            print('\n\n')
            #self.__print_helper(node.left, level)

    def get_splits(self):
        return self.__get_splits_helper(self.tree)
    
    def __get_splits_helper(self, node):
        if node is  None:
            return []
        else:
            res = []
            if node.f_idx != -1:
                res.append(node.val)
            res.extend(self.__get_splits_helper(node.right))
            res.extend(self.__get_splits_helper(node.left))
            return res


            