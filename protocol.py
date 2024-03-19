import itertools,random
import numpy as np
import base,data,exp,utils

class BasicProtocol(base.Protocol):
    def __init__(self,n_split=10,n_iters=10):
        self.n_split=n_split
        self.n_iters=n_iters
        self.exp_group=None
    
    def init_exp_group(self):
        all_exps=[[] for _ in range(self.n_iters)]
        self.exp_group=BasicExpGroup(all_exps=all_exps,
                                     n_split=self.n_split,
                                     n_iters=self.n_iters)

    def single_split(self,dataset):
        all_splits=gen_all_splits(self.n_split,dataset)
        train=list(itertools.chain.from_iterable(all_splits[:-1]))
        test=all_splits[-1]
        return base.Split(dataset=dataset,
                          train=train,
                          test=test)

    def iter(self,dataset):
        for i in range(self.n_iters):
            for train_i,test_i in self.gen_split(dataset):
                yield base.Split(dataset=dataset,
                                 train=train_i,
                                 test=test_i)

    def add_exp(self,exp_i):
        if(self.exp_group is None):
            self.init_exp_group()
        self.exp_group.add(exp_i)

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

class BasicExpGroup(object):
    def __init__(self,all_exps,n_split=10,n_iters=10):
        self.n_split=n_split
        self.n_iters=n_iters
        self.all_exps=all_exps 
        self.current_split=0
        self.current_iter=0

    def add(self,exp_i):
        self.all_exps[self.current_iter].append(exp_i)
        self.current_split+=1
        if(self.current_split>=self.n_split):
            self.current_split=0
            self.current_iter+=1

    def eval(self,alg_params,clf_type="RF"):
        acc=[]
        for exp_i in self.all_exps:
            for exp_j in exp_i:
                result_j=exp_j.eval(alg_params,"RF")
                acc.append(result_j.acc())
        return acc

    def save(self,out_path):
        utils.make_dir(out_path)
        for i,exp_split_i in  enumerate(self.all_exps):
            utils.make_dir(f'{out_path}/{i}')
            for j,exp_j in enumerate(exp_split_i):
                path_j=f'{out_path}/{i}/{j}'
                exp_j.save(path_j)
                np.save(f'{path_j}/test',exp_j.split.test)


def read_basic(in_path,dataset_path):
    dataset=data.get_data(dataset_path)
    all_exps=[]
    for path_i in utils.top_files(in_path):
        all_exps.append([])
        for exp_path_j in utils.top_files(path_i):
            exp_j= exp.read_exp(exp_path_j,dataset)
            all_exps[-1].append(exp_j)
        splits=[exp_i.split.test 
                     for exp_i in all_exps[-1]]
        for k,exp_k in enumerate(all_exps[-1]):
            train_k=get_train(k,splits)
            exp_k.split.train=train_k
    return BasicExpGroup(all_exps=all_exps,
                         n_split=len(all_exps),
                         n_iters=len(all_exps[0]))

def get_train(k,splits):
    train=[]
    for i,split_i in enumerate(splits):
        if(k!=i):
            train+=list(split_i)
    return np.array(train).astype(int)