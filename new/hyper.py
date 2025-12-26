from itertools import product
import base,dataset,tree_clf

class HyperparamSpace(object):
    def __init__( self,
                  extr=["info","ind","prod"],
                  n_feats=[10,20,30,50]):
        self.extr=extr
        self.n_feats=n_feats

    def __call__(self):
        for feat_i,dim_i in product(self.extr,self.n_feats):
            yield { "feat_type":feat_i,
                      "n_feats":dim_i}


def optim_hyper(in_path):
    data=dataset.read_csv(in_path)
    splits=base.get_splits(data,10,1)
    splits=enumerate(splits)
    hyper_space=HyperparamSpace()
    for hyper_i in hyper_space():
        factory_i=tree_clf.TreeFactory(hyper_i)
        result=factory_i.get_results(data,splits)
        print(result)

in_path="../incr_exp/uci/data/vehicle"
optim_hyper(in_path)