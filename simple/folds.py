import data

def make_folds(data_dict,k_folds=10):
    if(type(data_dict)==str):
        data_dict=data.read_data(data_dict)
    names=data_dict.names()
    folds=[[] for i in range(k_folds)]
    cats=names.by_cat()
    for cat_i in cats.values():
        cat_i.shuffle()
        for j,name_j in enumerate(cat_i):
            folds[j % k_folds].append(name_j)
    return folds

def get_splits(data_dict,folds):
    for i in range(len(folds)):
        test=folds[i]
        train=[]
        for j,fold_j in enumerate(folds):
            if(i!=j):
                train+=fold_j
        new_names={}
        for name_i in train:
            name_i=data.Name(name_i)
            new_names[name_i]=name_i.set_train(False)
        for name_i in test:
            name_i=data.Name(name_i)
            new_names[name_i]=name_i.set_train(True)
        yield data_dict.rename(new_names),new_names