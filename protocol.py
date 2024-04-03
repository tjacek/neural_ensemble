import itertools,random
import numpy as np
import base,data,exp,utils

class Protocol(object):
    def __init__(self,io_type,
                      split_gen=None,
                      alg_params=None):
        if(split_gen is None):
            split_gen=SplitGenerator(n_split=3,
                                     n_iters=3)
        if(alg_params is None):
            alg_params=base.AlgParams()
        self.io_type=io_type
        self.split_gen=split_gen
        self.alg_params=alg_params
       

    def get_group(self,exp_path:str):
        return ExpGroup(exp_path=exp_path,
                         io_type=self.io_type,
                         n_split=self.split_gen.n_split,
                         n_iters=self.split_gen.n_iters)

class SplitGenerator(object):
    def __init__(self,n_split=10,n_iters=10):
        self.n_split=n_split
        self.n_iters=n_iters

    def single_split(self,dataset):
        all_splits=gen_all_splits(self.n_split,dataset)
        train=list(itertools.chain.from_iterable(all_splits[:-1]))
        test=all_splits[-1]
        return base.Split(dataset=dataset,
                          train=train,
                          test=test)

    def iter(self,dataset):
        for i in range(self.n_iters):
            for j,(train_j,test_j) in enumerate(self.gen_split(dataset)):
                split_ij=base.Split(dataset=dataset,
                                    train=train_j,
                                    test=test_j)
                yield i,j,split_ij

    def gen_split(self,dataset):
        all_splits=gen_all_splits(self.n_split,dataset)
        for i in range(self.n_split):
            train_i=[]
            for j in range(self.n_split):
                if(i!=j):
                    train_i+=all_splits[j]
            test_i=all_splits[i]
            yield train_i,test_i


def gen_all_splits(n_split,dataset):
    by_cat=dataset.by_cat()
    all_splits=[[] for i in range(n_split) ]
    for cat_i,samples_i in by_cat.items():
        random.shuffle(samples_i)
        for j,index in enumerate(samples_i):
            mod_j=(j%(n_split))
            all_splits[mod_j].append(j)
    return all_splits

class ExpGroup(object):
    def __init__(self,exp_path:str,
                      io_type,
                      n_split=10,
                      n_iters=10):
        self.exp_path=exp_path
        self.io_type=io_type
        self.n_split=n_split
        self.n_iters=n_iters

    def init_dir(self):
        utils.make_dir(self.exp_path)
        for i in range(self.n_split):
            utils.make_dir(f'{self.exp_path}/{i}')
    
    def iter(self,dataset):
        for i in range(self.n_iters):
            for j in range(self.n_split):
                yield self.get(dataset,i,j)

    def set(self,exp_ij,i,j):
        path_ij=f'{self.exp_path}/{i}/{j}'
#        exp_ij.save(path_ij)
        self.io_type.save(exp_ij,path_ij)
        np.save(f'{path_ij}/test',exp_ij.split.test)

    def get(self,dataset,i,j):
        path_ij=f'{self.exp_path}/{i}/{j}'
        exp_ij= exp.read_exp(path_ij,dataset)
        exp_ij.split.train=self.get_train(i,j)
        return exp_ij

    def get_train(self,i,j):
        train=[]
        for k in range(self.n_iters):
            if(k!=j):
                path_ik=f'{self.exp_path}/{i}/{k}'
                train_k=np.load(f'{path_ik}/test.npy')
                train+=list(train_k)
        return np.array(train).astype(int)

def read_facade(in_path:str):
    paths=utils.top_files(in_path)
    n_iters=len(paths)
    n_split=len(utils.top_files(paths[0]))
    return ExpFacade(exp_path=in_path,
                     n_split=n_iters,
                     n_iters=n_split)