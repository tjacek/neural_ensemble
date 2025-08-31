import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class Nodes(object):
    def __init__( self,
                  thres,    
                  feats,
                  dist,
                  cols ):
        self.thres=thres
        self.feats=feats
        self.dist=dist
        self.cols=cols

    def print(self):
        for i,feat_i in enumerate(self.feats):
            col_i=self.cols[feat_i]
            print(f"{col_i},{self.thres[i]:.4f}")

def parse_nodes(clf,cols):
    index=[ i for i,feat_i in enumerate(clf.tree_.feature)
              if(feat_i>=0)]
    thres,feats,dist=[],[],[]
    for i in index:
        thres.append(clf.tree_.threshold[i])
        feats.append(clf.tree_.feature[i])
        dist.append(clf.tree_.value[i])
    return Nodes(thres,feats, dist,cols)

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
    nodes.print()
    tree.plot_tree(clf,feature_names=cols)
    plt.show()

in_path="../multi_exp/data/car"
cols=[ 'buying','maint','doors', 
       'persons', 'lug_boot', 'safety' ]
exp(in_path,cols)
