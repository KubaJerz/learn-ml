from d_tree import DecisionTree
import numpy as np
import pandas as pd
import random

class RandomForest():

    def __init__(self, num_trees = 10, num_rand_features = 2, max_depth = 5, min_samples = 1, min_info_gain = 0.0, criterion = 'gini') -> None:
        self.num_trees = num_trees #number of trees to have in the forest
        self.num_rand_features = num_rand_features #the number of random features to concider when making split. so if n_r_f = 5 it will only concider 5 out of all time random features


        self.max_depth = max_depth #mac depth the tree can be
        self.min_samples = min_samples #min samples allowed in leaf for us to continues to split on it 
        self.min_info_gain = min_info_gain #min info gain needed to continue to build the tree
        self.criterion = criterion

    def __build_trees(self):
        self.trees = []

        i = 0
        while i < self.num_trees:
            self.trees.append(DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples, min_info_gain=self.min_info_gain, criterion=self.criterion, num_rand_features=self.num_rand_features))
            i += 1

    def bootstrap(self, X_train: np.array, y_train: np.array):
        X_new = np.zeros(X_train.shape)
        y_new = np.zeros(y_train.shape)

        for i in range(len(X_train)):
            rand_idx = random.randint(0, (len(X_train)-1))
            X_new[i] = X_train[rand_idx]
            y_new[i] = y_train[rand_idx]
        
        return X_new, y_new

    def fit(self, X_train, y_train):
        #build trees
        self.__build_trees()

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.to_numpy()
              
        #fit trees
        for tree in self.trees:
            X_train, y_train = self.bootstrap(X_train, y_train)
            tree.fit(X_train, y_train)


    def pred(self, X: np.array) -> np.array: 
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        preds = []
        for x in X:
            pred = []
            for tree in self.trees:
                pred.append(tree.predict_one_sample(x))
            preds.append(max(pred, key=pred.count))
        
        return preds
        

        

