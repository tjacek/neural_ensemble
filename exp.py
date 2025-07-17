import clfs,dataset

def exp(in_path):
    data_i=dataset.read_csv(in_path)
    tree_i=clfs.get_clfs("TREE-MLP")()
    tree_i.fit(data_i.X,data_i.y)
    result=data_i.pred(None,tree_i.clf)
    print(result.get_acc())

exp("bad_exp/data/wine-quality-red")