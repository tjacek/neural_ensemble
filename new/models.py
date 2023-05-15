import numpy as np
import clfs,tools

class ManyClfs(object):
    def __init__(self,dir_path):
        tools.make_dir(dir_path)
        self.dir_path=dir_path

    def read(self):
        for path_i in tools.top_files(self.dir_path):
            yield single_read(path_i)

    def split(self,X,y):
        for i,(model_i,split_i) in enumerate(self.read()):
            train_i,test_i=split_i
            X_train,y_train=X[train_i],y[train_i]
            X_test,y_test=X[test_i],y[test_i]
            yield i,model_i,(X_train,y_train),(X_test,y_test)

    def save(self,clfs_dict,i,split_i):
        split_dir=f'{self.dir_path}/{i}'
        tools.make_dir(split_dir)
        np.save(f'{split_dir}/train',split_i[0])
        np.save(f'{split_dir}/test',split_i[1])
        for name_j,clf_j in clfs_dict.items():
            clfs.save_clf(clf_j,f'{split_dir}/{name_j}')

def split_iterator(cv,X,y):
    for i,(train_i,test_i) in enumerate(cv.split(X,y)):
        X_train,y_train=X[train_i],y[train_i]
        X_test,y_test=X[test_i],y[test_i]
        yield i,(train_i,test_i),(X_train,y_train),(X_test,y_test)

def single_read(path_i):
    clf_i= { 
            model_path.split('/')[-1]:clfs.read_clf(model_path)
                for model_path in tools.get_dirs(path_i)}
    train_i=np.load(f'{path_i}/train.npy')
    test_i=np.load(f'{path_i}/test.npy')
    return clf_i,(train_i,test_i)