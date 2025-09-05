import numpy as np
from collections import Counter,defaultdict
from scipy.stats import entropy 
from sklearn import tree
import utils

def get_tree(tree_type):
    if(tree_type=="random"):
        return RandomTree()
    return GradientTree()

class GradientTree(object):
    def __call__(self):
        return tree.DecisionTreeClassifier(max_depth=3,
                                       class_weight="balanced")

    def __str__(self):
        return "GradientTree"

class RandomTree(object):
    def __call__(self):
        return tree.DecisionTreeClassifier(max_features='sqrt')
#                                           class_weight="balanced")

    def __str__(self):
        return "RandomTree"

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

class NodeDesc(object):
    def __init__( self,
                  parent,
                  feature,
                  threshold,
                  right_dist,
                  n_samples):
#                  left_dist):
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

def inf_features(tree_dict,n_feats=10):
    info,nodes=tree_dict.mutual_info()
    info_index=np.argsort(info)[:10]
    s_nodes=[nodes[i] for i in info_index]
#    for i in s_nodes:
#        s=tree_dict[i]
#        print(s.right_dist)
#    raise Exception(s_nodes)
    return s_nodes

def log_helper(p_target_node, p_node):
    if(p_node==0):
        return 0
    total=0
    for i,p_i in enumerate(p_target_node):
        if(p_i==0):
            continue
        total+= p_i*np.log(p_i/p_node)
    return -total

class TabFeatures(object):
    def __call__(self,X,concat=True):
        new_feats=[self.compute_feats(x_i) for x_i in X]
        new_feats=np.array(new_feats)
        if(concat):
            return np.concatenate([X,new_feats],axis=1)
        return new_feats 

    def compute_feats(self,x_i):
        raise NotImplementedError()

class TreeFeatures(TabFeatures):
    def __init__(self,features,thresholds):
        self.features=features
        self.thresholds=thresholds

    def n_feats(self):
        return len(self.features)

    def compute_feats(self,x_i):
        new_feats=[]
        for i,feat_i in enumerate(self.features):
            value_i=x_i[feat_i]
            thres_i=self.thresholds[i]
            new_feats.append(int(value_i<thres_i) )
        return np.array(new_feats)

    def save(self,out_path):
        utils.make_dir(out_path)
        np.save(f"{out_path}/feats.npy",self.features)
        np.save(f"{out_path}/thresholds.npy",self.thresholds)

def read_feats(in_path):    
    feats=np.load(f"{in_path}/feats.npy")
    thres=np.load(f"{in_path}/thresholds.npy")
    return TreeFeatures(feats,thres)