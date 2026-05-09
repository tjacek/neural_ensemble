import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy 
import pandas as pd
import argparse

def exp(in_path,cols):
    X,y=read_data(in_path,cols)
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X,y)
    tree_dict=make_tree_dict(clf)
    info,nodes=tree_dict.mutual_info()
    print(cols)
    print("feature,threshold,mutual_info")
    for info_i,node_i in zip(info,nodes):
        desc_i=tree_dict[node_i].get_desc(cols)

        print(f"{desc_i},{info_i:.4f}")
    for path_i in tree_dict.indv_paths(nodes):
        print(f"Path:{path_i}")
        path_i.reverse()
        for node_j in path_i:
        	print(tree_dict[node_j].get_desc(cols))
    tree.plot_tree(clf,feature_names=cols)
    plt.show()

class TreeDict(dict):
    def __init__(self, arg=[]):
        super(TreeDict, self).__init__(arg)
        self.targe_dist=None
        self.n_samples=None
    
    def get_attr(self,attr,s_nodes):
        return [ self[i](attr) for i in s_nodes]
    
    def mutual_info(self):
        h_y=entropy(self.target_dist)
        by_cat=self.target_dist*self.n_samples
        info,nodes=[],[]
        for i,node_i in self.items():
            p_target_1=node_i.right_dist
            p_target_0=node_i.get_dist(by_cat)
            p_1= node_i.get_p(self.n_samples)
            p_0=1.0-p_1
            h_node_y  =  log_helper(p_target_1, p_1)
            h_node_y +=  log_helper(p_target_0, p_0)
            i_xy= h_y - h_node_y
            info.append(i_xy)
            nodes.append(i)
        return info,nodes

    def get_paths(self,s_nodes):
        indexes=[]
        for path_j in self.indv_paths(s_nodes):     
            indexes+=path_j 
        return list(set(indexes))

    def indv_paths(self,s_nodes):
        for i in s_nodes:
            node_i=self[i]
            path_j=[i]
            current=node_i
            while(current.parent!= (-2)):
                path_j.append(current.parent)
                current=self[current.parent]
            yield path_j

class NodeDesc(object):
    def __init__( self,
                  parent,
                  feature,
                  threshold,
                  right_dist,
                  n_samples):
        self.parent=parent
        self.feature=feature
        self.threshold=threshold
        self.right_dist=right_dist
        self.n_samples=n_samples
    
    def get_dist(self,by_cat):
        dist_i=self.right_dist*self.n_samples
        diff_i=by_cat-dist_i
        diff_i/=np.sum(diff_i)
        return diff_i

    def get_p(self,total_samples):
        return self.n_samples/total_samples

    def __call__(self,attr):
        if(attr=="threshold"):
            return self.threshold
        if(attr=="feat"):
            return self.feature

    def get_desc(self,cols):
        desc_i=cols[self.feature]
        return f"{desc_i},{self.threshold:.4f}"

def read_data(in_path,cols):
    df=pd.read_csv(in_path,header=None)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    return X,y

def make_tree_dict(clf):
    tree_dict=TreeDict()
    raw_tree=clf.tree_
    parent=find_params(raw_tree)    
    root=np.argmin(parent)
    tree_dict.target_dist= raw_tree.value[root][0] 
    tree_dict.n_samples=raw_tree.weighted_n_node_samples[root]
    for i,parent in enumerate(parent):
        if(parent!=(-1)):
            right_i=raw_tree.children_right[i]
            right_dist=raw_tree.value[right_i][0]
            n_samples=raw_tree.weighted_n_node_samples[right_i]
            node_i=NodeDesc(parent=parent,
                            feature=raw_tree.feature[i],
                            threshold=raw_tree.threshold[i],
                            right_dist=right_dist,
                            n_samples=n_samples)
            tree_dict[i]=node_i            
    return tree_dict

def find_params(raw_tree):
    n_nodes=len(raw_tree.feature)
    parent= -2*np.ones((n_nodes,),dtype=int)
    for i in range(n_nodes):
        left_i=raw_tree.children_left[i]
        right_i=raw_tree.children_right[i]
        if(left_i<0 or right_i <0):
            parent[i]= -1
            continue
        parent[left_i]=  i
        parent[right_i]= i
    return parent

def log_helper(p_target_node, p_node):
    if(p_node==0):
        return 0
    total=0
    for i,p_i in enumerate(p_target_node):
        if(p_i==0):
            continue
        quot_i=p_i/p_node
        if(quot_i<=0):
            continue
        total+= p_i*np.log(quot_i)
    return -total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="car")
    parser.add_argument("--names", type=str, default="car_names")
    args = parser.parse_args()
    with open(args.names, 'r') as file:
         cols = file.readlines()
         cols=[ col_i.strip() for col_i in cols]
    exp(args.data,cols)