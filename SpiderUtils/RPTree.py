import numpy as np
from sklearn import random_projection
from sklearn.decomposition import PCA
import random

class Node(object):
    def __init__(self, data):
        self.data           = data
        self.index          = None
        self.hyperplane     = None
        self.PCAmean        = None
        self.splitDimension = None
        self.splitPoint     = None
        self.left           = None
        self.right          = None

    def get_leaf_nodes(self):
        leaves = []
        self._collect_leaf_nodes(self,leaves)
        return leaves

    def _collect_leaf_nodes(self, node, leaves):
        if node is not None:
            if node.left==None and node.right==None:
                leaves.append(node.index)
                
            self._collect_leaf_nodes(node.left, leaves)
            self._collect_leaf_nodes(node.right, leaves)

class BinaryTree(object):
    def __init__(self, root):
        self.root = Node(root)
      
    def construct_tree(self, tree, index):
        nTry = 3 
        X_data = tree.root.data
        dispersion = 0
        for r in range(nTry):
            transformer = random_projection.GaussianRandomProjection(n_components=X_data.shape[1]-1)
            X_proj_temp = transformer.fit_transform(X_data)
            dispersionCurrent = np.max(np.std(X_proj_temp, axis=0))

            if  dispersionCurrent > dispersion:
                dispersion = dispersionCurrent
                X_proj = X_proj_temp
                hyperplane = transformer.components_
        
        SplitDimension = np.argmax(np.std(X_proj, axis=0))
        SplitPoint = random.uniform(np.quantile(X_proj[:,SplitDimension], 0.25, axis=0),np.quantile(X_proj[:,SplitDimension], 0.75, axis=0))
        
        X_left = X_data[np.where(X_proj[:,SplitDimension] < SplitPoint)[0]]
        X_left_index = index[np.where(X_proj[:, SplitDimension] < SplitPoint)[0]]
        X_right = X_data[np.where(X_proj[:,SplitDimension] >= SplitPoint)[0]]
        X_right_index = index[np.where(X_proj[:, SplitDimension] >= SplitPoint)[0]]

        tree.root.hyperplane = hyperplane
        tree.root.splitDimension = SplitDimension
        tree.root.splitPoint = SplitPoint
        
        if X_left.shape[0]>20:
            tree.root.left = self.construct_tree(BinaryTree(X_left), X_left_index)
        else:
            tree.root.left = Node(X_left)
            tree.root.left.index = X_left_index
            
        if X_right.shape[0]>20:
            tree.root.right = self.construct_tree(BinaryTree(X_right), X_right_index)
        else:
            tree.root.right = Node(X_right)
            tree.root.right.index = X_right_index
            
        return tree.root                                       
            
