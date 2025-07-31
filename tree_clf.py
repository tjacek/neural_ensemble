import numpy as np
import tree_feats

class TreeDict(dict):
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
    print(tree_dict["samples"])
    print(dir(clf.tree_))
    
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