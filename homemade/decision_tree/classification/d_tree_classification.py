from collections import Counter
import math
import numpy as np
from tree_node_class import TreeNode
import pandas as pd

class DecisionTree():

    def __init__(self, max_depth = 5, min_samples = 1, min_info_gain = 0.0, criterion = 'gini', num_rand_features = -1) -> None:
        self.max_depth = max_depth #mac depth the tree can be
        self.min_samples = min_samples #min samples allowed in leaf for us to continues to split on it 
        self.min_info_gain = min_info_gain #min info gain needed to continue to build the tree
        self.labels = None
        self.criterion = criterion
        self.num_rand_features = num_rand_features #the number of random features to concider when making split. so if n_r_f = 5 it will only concider 5 out of all time random features


    def __entrpoy(self, all_labels: list) -> float:
        class_prob = [label_count / len(all_labels) for label_count in Counter(all_labels).values()] #this well calc the prob each class being randomly setlected so len(set) = 10, five 0's, five 1's then class_prob = [.5,.5]
        return -1 * sum(p * math.log(p,2) for p in class_prob)
    
    def __gini(self, all_labels: list) -> float:
        class_prob = [label_count / len(all_labels) for label_count in Counter(all_labels).values()]
        return (1 - sum(prob ** 2 for prob in class_prob))
    

    '''
    This will return the sum weighted entropy for two lists
    '''
    def __sets_entropy(self, splits: list) -> float:
        total = sum(len(split) for split in splits)
        return sum(len(split)/total * self.__entrpoy(split) for split in splits) # we get the entropy of each split then weight it by its size compared to the total size of the splits then sum

    def __sets_gini(self, splits: list) -> float:
        total = sum(len(split) for split in splits)
        return sum(len(split)/total * self.__gini(split) for split in splits)

    def __split(self, data, f_index, threashold):
        bool_mask = data[:, f_index] < threashold #this makes array where for each row it 0 or 1 if if meet or did not the threashhold
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
                        if criterion == 'entropy':
                            split_gain = self.__sets_entropy([data_left[:, -1], data_right[:, -1]])
                        else:
                            split_gain = self.__sets_gini([data_left[:, -1], data_right[:, -1]])

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
        criterion = self.criterion

        if curr_depth >= self.max_depth:
            return None

        s_data_left, s_data_right, s_f_idx, s_f_val, split_entropy_or_gini = self.__find_best_split(data) #find the ebst split and info form the split

        if criterion == 'entropy':
            curr_entropy = self.__entrpoy(data[:, -1]) #gets the entropy of the curr data 
            info_gain = curr_entropy - split_entropy_or_gini #calc the information gain from the best split on this data
        else:
            curr_gini = self.__gini(data[:, -1]) #gets the gini of the curr data 
            info_gain = curr_gini - split_entropy_or_gini #calc the information gain from the best split on this data

        node = TreeNode(data, s_f_idx, s_f_val, info_gain)

        if info_gain <= 0 or info_gain < self.min_info_gain:
            return TreeNode(data, s_f_idx, s_f_val, info_gain)

        elif self.min_samples >= s_data_left.shape[0] or self.min_samples >= s_data_right.shape[0]:
            return TreeNode(data, s_f_idx, s_f_val, info_gain)
        
        curr_depth += 1
        node.left = self.__build_tree(s_data_left, curr_depth)
        node.right = self.__build_tree(s_data_right, curr_depth)

        return node
    
    '''
    trains the tree
    '''
    def fit(self, X_train: np.array, y_train: np.array):
        train_data = np.concatenate((X_train, np.reshape(y_train,(-1,1))),axis = 1)

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
            return tree.label
        split_index = tree.f_idx
        if x[split_index] < tree.f_val:
            return self.predict_helper(x, tree.left)
        else:
            return self.predict_helper(x, tree.right)
        
        
    def set_labels(self, lables):
        self.labels = lables
        
    def print(self):
        level = 0
        print('    '*3,'level: ',level,end='     ')
        self.tree.print(self.labels)
        print('\n')
        self.__print_helper(self.tree, level)


    def __print_helper(self, node: TreeNode, level: int):
        if node.left == None or node.right == None:
            return
        else:
            level += 1
            print('Level: ',level,'  LEFT: ',end='')
            node.left.print(self.labels)
            print('\n')
            self.__print_helper(node.left, level)
            print('level: ',level,'  RIGHT:  ',end='')
            node.right.print(self.labels)
            print('\n')
            self.__print_helper(node.right, level)

            print('\n\n')
            #self.__print_helper(node.left, level)
            