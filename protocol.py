import itertools,random
import numpy as np
import base,data,deep,exp,utils

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
        return self.io_type(exp_path=exp_path,
                            n_split=self.split_gen.n_split,
                            n_iters=self.split_gen.n_iters)

    def __str__(self):
        return f'split:{self.split_gen}\nio_type{self.io_type}'

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
    
    def __str__(self):
        return f'{self.n_split}-{self.n_iters}'

def gen_all_splits(n_split,dataset):
    by_cat=dataset.by_cat()
    all_splits=[[] for i in range(n_split) ]
    for cat_i,samples_i in by_cat.items():
        random.shuffle(samples_i)
        for j,index in enumerate(samples_i):
            mod_j=(j%(n_split))
            all_splits[mod_j].append(j)
    return all_splits

class ExpIO(object):
    def __init__(self,exp_path:str,
                      n_split=10,
                      n_iters=10):
        self.exp_path=exp_path
        self.n_split=n_split
        self.n_iters=n_iters

    def init_dir(self):
        utils.make_dir(self.exp_path)
        for i in range(self.n_iters):
            utils.make_dir(f'{self.exp_path}/{i}')

    def iter_necscf(self,dataset):
        for i,j,path_ij in self.iter_paths():
            print(f'{i}/{j}')
            yield self.get_necscf(i,j,path_ij,dataset)    

    def iter_paths(self):
        for i in range(self.n_iters):
            for j in range(self.n_split):
                yield i,j,f'{self.exp_path}/{i}/{j}'

#    def iter_result(self,dataset):
#        for i,j,path_ij in self.iter_paths():
#            yield self.get_result(i,j,path_ij)

    def get_train(self,i,j):
        train=[]
        for k in range(self.n_iters):
            if(k!=j):
                path_ik=f'{self.exp_path}/{i}/{k}'
                train_k=np.load(f'{path_ik}/test.npy')
                train+=list(train_k)
        return np.array(train).astype(int)

    def set(self,exp_ij,i,j,dataset=None):
        path_ij=f'{self.exp_path}/{i}/{j}'
        self.save(exp_ij,path_ij,dataset)
        np.save(f'{path_ij}/test',exp_ij.split.test)

class NNetIO(ExpIO):

#    def iter_exp(self,dataset):
#        for i,j,path_ij in self.iter_paths():
#            yield self.get_exp(i,j,path_ij,dataset)    

    def get_exp(self,i,j,in_path,dataset):
        with open(f'{in_path}/info',"r") as f:
            lines=f.readlines()
            hyper_params=eval(lines[0])
            model=deep.ensemble_builder(dataset.params,hyper_params)
            train=self.get_train(i,j)
            test=np.load(f'{in_path}/test.npy')
            split=base.Split(dataset=dataset,
                            train=train,
                            test=test)
            return exp.Experiment(split=split,
                                hyper_params=hyper_params,
                                model=model)

    def get_necscf(self,i,j,path,dataset):
        exp_ij=self.get_exp(i,j,path,dataset)
        return exp_ij.to_necscf()
#        return exp_i.split.eval(clf_type)

    def save(self,exp,out_path,dataset):
        utils.make_dir(out_path)
        exp.model.save_weights(f'{out_path}/weights')
        with open(f'{out_path}/info',"a") as f:
            f.write(f'{str(exp.hyper_params)}\n') 

    def __str__(self):
        return "NNetIO"

class FeatIO(ExpIO):
    def get_necscf(self,i,j):
        cs=np.load(f'{in_path}/info')
        return split_i.ncscf_from_feats()

    def save(self,exp,out_path,dataset):
        utils.make_dir(out_path)
        extractor=exp.make_extractor()
        necscf=exp.split.to_ncscf(extractor)
        binary=np.array(extractor.predict(dataset.X))
        np.savez_compressed(f'{out_path}/info',binary)

    def __str__(self):
        return "FeatIO"

#class ResultIO(ExpIO):

#    def get_result(self,i,j,path,clf_type):
#        return base.read_result(path)

#    def save(self,exp,out_path):
#        utils.make_dir(out_path)
#        result=exp.eval(alg_params,clf_type="RF")
#        result.save(out_path)

#    def __str__(self):
#        return "ResultIO"