import base,clfs,dataset

def tree_exp(in_path):
    data=dataset.read_csv(in_path)
    tree=base.get_clf("TREE")
    tree.fit(data.X,data.y)
    tree_repr=clfs.make_tree_features(tree)
    new_data=dataset.Dataset(X=tree_repr(data.X,concat=False),
                             y=data.y)
    split_k=base.random_split(new_data)
    result,_=split_k.eval(new_data,"LR")
    print(result.get_acc())
    print(new_data.n_cats())

def simple_exp(in_path):
    data_i=dataset.read_csv(in_path)
    split_k=base.random_split(data_i)
    clf=base.get_clf(clf_type="RF")
    result,_=split_k.eval(data_i,clf)
    print(result.get_acc())

def nn_exp(in_path): 
    data=dataset.read_csv(in_path)
    clf_factory=clfs.get_clfs(clf_type="TREE-MLP")
    clf_factory.init(data)
    nn_clf=clf_factory()
    split_k=base.random_split(data)
    result,_=split_k.eval(data,nn_clf)
    print(result.get_acc())

simple_exp("bad_exp/data/wine-quality-red")