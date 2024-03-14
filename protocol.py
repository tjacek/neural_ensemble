import itertools,random
import base

class BasicProtocol(base.Protocol):
    def __init__(self,n_split=10,n_iters=10):
        self.n_split=n_split
        self.n_iters=n_iters
        self.exp_group=None

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
            self.exp_group=BasicExpGroup(self.splits,self.n_iters)
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
            mod_j=(j%n_split)
            all_splits[mod_j].append(j)
    return all_splits

class BasicExpGroup(object):
    def __init__(self,n_split=10,n_iters=10):
        self.n_split=n_split
        self.n_iters=n_iters
        self.all_exps=[[] for _ in range(n_iters)]
        self.current_split=0
        self.current_iter=0

    def add(self,exp_i):
        self.all_exps[self.current_iter].append(exp_i)
        self.current_split+=1
        if(self.current_split >self.n_split):
            self.current_split=0
            self.current_iter+=1