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
    def __len__(self):
        return len(self["feat"])

    def get_node(self,i):
        keys=list(self.keys())
        keys.sort()
        print(keys)
        if(type(i)!=list):
            return [self[key_j][i] for key_j in keys]
        return [[self[key_j][k] for key_j in keys]
                         for k in i]
    
    def get_attr(self,key,nodes):
        return [self[key][i] for i in nodes]

    def mutual_info(self):
        dist=self["value"][0]
        n_samples=len(self)
        offset=np.ones(dist.shape)
        h_y=entropy(dist)
        mi=[]
        for i,value_i in enumerate(self["value"]):
            samples_i=self["samples"][i]
            value_i=self["value"][i]
            p_y= samples_i/ n_samples
            h_yx=p_y*entropy(value_i)#,nan_policy='omit')
            h_yx+=(1-p_y)*entropy(offset-value_i)
            i_xy=h_y-h_yx
            mi.append(i_xy)
        return mi

    def get_path(self,i):
        path=[i]
        while(i>=0):
            i=self["parent"][i]
            path.append(i)
        return path

def make_tree_dict(clf):
    tree_dict=TreeDict()
    tree_dict["threshold"]=clf.tree_.threshold
    tree_dict["feat"]=clf.tree_.feature
    tree_dict["left"]=clf.tree_.children_left
    tree_dict["right"]=clf.tree_.children_right
    tree_dict["value"]=[ value_i.flatten() 
                            for value_i in clf.tree_.value]
    tree_dict["samples"]=clf.tree_.weighted_n_node_samples
    n_nodes=len(tree_dict["feat"])
    tree_dict["parent"]= -np.ones((n_nodes,),dtype=int)
    for i in range(n_nodes):
        left_i=tree_dict["left"][i]
        right_i=tree_dict["right"][i]
        if(left_i>=0):
            tree_dict["parent"][left_i]=i
            tree_dict["parent"][right_i]=i
    return tree_dict

def inf_features(tree_dict,n_feats=5):
    mutual_info=tree_dict.mutual_info()
    index=np.argsort(mutual_info)
    s_feats=[]
    for i in index[:n_feats]:
        s_feats+=tree_dict.get_path(i)
    s_feats=set(s_feats)
    s_feats.remove(-1)
    s_feats=list(set(s_feats))
    return s_feats

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

class DiscFeats(TabFeatures):
    def __init__(self,thres_dict):
        self.thres_dict=thres_dict
    
    def compute_feats(self,x):
        new_feats=[]
        for feat_i,x_i in enumerate(x):
            if(feat_i in self.thres_dict):
                thres_i=self.thres_dict[feat_i]
                value_i=None
                for j,thres_j in enumerate(thres_i):
                    if(x_i<thres_j):
                        value_i=j
                        break
                if(value_i is None):
                    value_i=len(thres_i)
                new_feats.append(value_i)
        return new_feats

def make_disc_feat(feats,thresholds):
    thres_dict=defaultdict(lambda:[])
    for i,feat_i in enumerate(feats):
        if(feat_i):
            thres_i=thresholds[i]
            thres_dict[feat_i].append(thres_i)
    new_dict={}
    for feat_i,thres_i in thres_dict.items():
        thres_i.sort()
        thres_i=np.array(thres_i)
        new_dict[feat_i]=np.array(thres_i)
    return DiscFeats(new_dict)

class IndFeatures(TabFeatures):
    def __init__(self,nodes,tree):
        self.nodes=nodes
        self.tree=tree

    def __call__(self,X,concat):
        ind_vars=self.tree.decision_path(X)
        new_X=[ind_vars[:,i].toarray() 
                    for i in self.nodes]
        new_X=np.concatenate(new_X,axis=1)       
        if(concat):
            new_X=np.concatenate([X,new_X],axis=1)
        return new_X

class ConcatFeatures(TabFeatures):
    def __init__(self,indiv_extr):
        self.indiv_extr=indiv_extr

    def __call__(self,X,concat=True):
        new_X=[indv_i(X) for indv_i in self.indiv_extr]
        new_X=np.concatenate(new_X,axis=1)
        if(concat):
            return np.concatenate([X,new_X],axis=1)
        return new_X

if __name__ == '__main__':
    import base,dataset
    data=dataset.read_csv("bad_exp/data/wine-quality-red")
    data.y= data.y.astype(int)
    clf=get_tree("random")()
    clf.fit(data.X,data.y)
    indexes=inform_nodes(clf,data.y)
    show_nodes(clf,indexes[:20])
#    tree.plot_tree(clf, proportion=True)
#    plt.show()
#    thres_feat=make_thres_feats(clf)
#    thres_feat.group()
#    new_X=thres_feat(data.X,concat=False)
#    thre_stats(new_X,data.y)
