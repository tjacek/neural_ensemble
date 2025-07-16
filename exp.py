import clfs,dataset

def exp(in_path):
    data_i=dataset.read_csv(in_path)
    tree_i=clfs.get_clfs("TREE")()
    print(dir(tree_i.clf))

exp("bad_exp/data/wine-quality-red")