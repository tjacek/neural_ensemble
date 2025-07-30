import numpy as np
from collections import Counter,defaultdict
from sklearn import tree

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

def make_tree_feats(tree):
    raw_tree=tree.tree_.__getstate__()['nodes']
    feats,thres=[],[]
    for node_i in raw_tree:
        feat_i=node_i[2]
        if(feat_i>=0):
            feats.append(feat_i)
            thres.append(node_i[3])
    return TreeFeatures(feats,thres)

class ThresholdFeats(TabFeatures):
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

    def propor(self):
        keys=list(self.thres_dict.keys())
        keys.sort()
        for key_i in keys:
            thres_i=self.thres_dict[key_i]
            thres_i-=thres_i[0]
            thres_i/=thres_i[-1]
            thres_i=np.round(thres_i,4)
            print(thres_i)

    def group(self,eps=0.05):
        new_thres={}
        for feat_i,thres_i in self.thres_dict.items():
            delta_i= np.abs(eps*(thres_i[-1]-thres_i[0]))
            diff_i=np.diff(thres_i)
            indexes=[j for j,diff_j in enumerate(diff_i)
                         if(np.abs(diff_j)>delta_i)]
            new_thres_i=[thres_i[j] for j in indexes]
            new_thres[feat_i]=np.array(new_thres_i)
        self.thres_dict= new_thres

def make_thres_feats(tree):
    raw_tree=tree.tree_.__getstate__()['nodes']
    thres_dict=defaultdict(lambda:[])
    for node_i in raw_tree:
        feat_i=node_i[2]
        if(feat_i>=0):
            thres_dict[feat_i].append(node_i[3])
    new_dict={}
    for feat_i,thres_i in thres_dict.items():
        thres_i.sort()
        thres_i=np.array(thres_i)
        new_dict[feat_i]=thres_i
    return ThresholdFeats(new_dict)

def thre_stats(X,y):
    y=[int(y_i) for y_i in y]
    n_cats=int(np.amax(y)+1)
    cat_sizes=Counter(y)
    for feat_i in X.T:
        n_thres=np.unique(feat_i).shape[0]
        hist_i=np.zeros((n_thres,n_cats))
        for j,cat_j in enumerate(y):
            value_j=feat_i[j]
            hist_i[value_j][cat_j]+=1
        for cat_i,size_i in cat_sizes.items():
            hist_i[:,cat_i]/=size_i
#            print(hist_i[:,cat_i])
        hist_i=np.round(hist_i,2)
        print(hist_i.T)

def inform_nodes(clf,y):
    cls_dist=get_disc_dist(y)
    div,desc=[],[]
    for i,value_i in enumerate(clf.tree_.value):
        n_samples=clf.tree_.weighted_n_node_samples[i]
        kl_i=KL(value_i,cls_dist)
        div.append(kl_i*np.log(n_samples))
        desc.append((n_samples,value_i))
    indexes=np.argsort(div)
    for i in indexes[:10]:
        print(div[i])
        n_samples,value=desc[i]
        print(n_samples)
        print(value)

def get_disc_dist(y):
    cat_sizes=Counter(y)
    keys=list(cat_sizes.keys())
    keys.sort()
    cls_dist=np.array([cat_sizes[key_i] 
                        for key_i in keys],
                        dtype=float)
    cls_dist/=np.sum(cls_dist)
    return cls_dist    

def KL(x,y):
    return np.sum(np.where(x != 0, x * np.log(x / y), 0))

def path_stat(clf,data):
    prob=[]
    for i in range(data.n_cats()):
        data_i=data.selection(data.y==i)
        path_i=clf.decision_path(data_i.X)
        prob_i= path_i.sum(axis=0)/len(data_i)
        prob.append(prob_i)
    prob=np.array(prob)
    for p in prob.T:
        print(p)
#    raise Exception(prob_i)

if __name__ == '__main__':
    import base,dataset
    data=dataset.read_csv("bad_exp/data/wine-quality-red")
    data.y= data.y.astype(int)
    clf=get_tree("random")()
    clf.fit(data.X,data.y)
    inform_nodes(clf,data.y)
#    tree.plot_tree(clf, proportion=True)
#    plt.show()
#    thres_feat=make_thres_feats(clf)
#    thres_feat.group()
#    new_X=thres_feat(data.X,concat=False)
#    thre_stats(new_X,data.y)
