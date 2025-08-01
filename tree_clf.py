import numpy as np
from scipy.stats import entropy 
import tree_feats

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
        dist=tree_dict["value"][0]
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
            print(type(i))
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


def inf_features(tree_dict):
    mutual_info=tree_dict.mutual_info()
    index=np.argsort(mutual_info)
    print(tree_dict.get_node(index[0]))
    print(tree_dict.get_path(index[0]))

if __name__ == '__main__':
    import base,dataset
    data=dataset.read_csv("bad_exp/data/wine-quality-red")
    data.y= data.y.astype(int)
    clf=tree_feats.get_tree("random")()
    clf.fit(data.X,data.y)
#    raise Exception(clf.tree_.__getstate__()['nodes'])
    tree_dict=make_tree_dict(clf)
    inf_features(tree_dict)
#    print(tree_dict.get_node(index[:10])[0])
#    print(tree_dict.get_attr("feat",index[:10]))