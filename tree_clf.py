import numpy as np
import tree_feats

class TreeDict(dict):

    def get_node(self,i):
        keys=self.keys()
        keys.sort()
        if(type(i)==int):
            return [self[key_j][i] for key_j in keys]
        return [[self[key_j][k] for key_j in keys]
                         for k in i]
    def show(self,i):
        value_i=clf.tree_.value[i]
        samples_i=clf.tree_.weighted_n_node_samples[i]
        print(samples_i)
        print(value_i)

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
    tree_dict["parents"]= -np.ones((n_nodes,))
    for i in range(n_nodes):
        left_i=tree_dict["left"][i]
        right_i=tree_dict["right"][i]
        if(left_i>=0):
            tree_dict["parents"][left_i]=i
            tree_dict["parents"][right_i]=i
    return tree_dict

if __name__ == '__main__':
    import base,dataset
    data=dataset.read_csv("bad_exp/data/wine-quality-red")
    data.y= data.y.astype(int)
    clf=tree_feats.get_tree("random")()
    clf.fit(data.X,data.y)
#    raise Exception(clf.tree_.__getstate__()['nodes'])
    make_tree_dict(clf)
#    facade=TreeFacade(clf)
#    print(facade)