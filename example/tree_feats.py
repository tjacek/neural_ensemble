import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import pandas as pd

class Nodes(object):
    def __init__( self,
                  thres,    
                  feats,
                  dists,
                  samples,
                  cols ):
        self.thres=thres
        self.feats=feats
        self.dists=dists
        self.samples=samples
        self.cols=cols

    def mutual_info(self,dist,y):
        offset=np.ones(dist.shape)
        h_y=entropy(dist)
        mi=[]
        for i,dist_y_i in enumerate(self.dists):
            p_node=self.samples[i]/len(y)
#            dist_node=np.array([p_node,1.0-p_node])
#            h_node=entropy(dist_node)
            h_node_y=    log_helper(dist_y_i, p_node)
            h_node_y  +=  log_helper(offset-dist_y_i, 1.0-p_node)
            i_xy= h_y - h_node_y
            print(i_xy)
            mi.append(i_xy)
        return mi
    
    def get_rep(self):
        for i,feat_i in enumerate(self.feats):
            yield f"{self.cols[feat_i]},{self.thres[i]:.4f}"

    def print(self):
        for col_i in self.get_rep():
            print(col_i)

def compute_dist(y):
    count_dict=Counter(y)
    cats=list(count_dict.keys())
    cats.sort()
    dist=np.array([count_dict[cat_i] 
                     for cat_i in cats],
                   dtype=float)
    dist/=np.sum(dist)
    return dist

def entropy(p):
    h=0.0
    for p_i in p:
        if(p_i==0):
            continue
        h+= p_i*np.log(p_i)
    return -h

def log_helper(p_target_node, p_node):
    if(p_node==0):
        return 0
    total=0
    for i,p_i in enumerate(p_target_node):
        if(p_i==0):
            continue
        total+= p_i*np.log(p_i/p_node)
    return -total

def parse_nodes(clf,cols):
    index=[ i for i,feat_i in enumerate(clf.tree_.feature)
              if(feat_i>=0)]
    thres,feats,dist,samples=[],[],[],[]
    for i in index:
        thres.append(clf.tree_.threshold[i])
        feats.append(clf.tree_.feature[i])
        dist.append(clf.tree_.value[i][0])
        samples.append(clf.tree_.weighted_n_node_samples[i])
    return Nodes(thres,feats, dist,samples,cols)

def read_data(in_path,cols):
    df=pd.read_csv(in_path,header=None)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    return X,y

def exp(in_path,cols):
    X,y=read_data(in_path,cols)
    clf = DecisionTreeClassifier(#criterion="entropy",
                                 max_depth=4)
    clf.fit(X,y)
    nodes=parse_nodes(clf,cols)
    full_dist=compute_dist(y)
    info=nodes.mutual_info(full_dist,y)
    for i,desc_i in enumerate(nodes.get_rep()):
        print(f"{desc_i},{info[i]:.4f}")
    tree.plot_tree(clf,feature_names=cols)
    plt.show()

in_path="../multi_exp/data/car"
cols=[ 'buying','maint','doors', 
       'persons', 'lug_boot', 'safety' ]
exp(in_path,cols)
