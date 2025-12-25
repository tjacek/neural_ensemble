from itertools import product

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
    hyper_space=HyperparamSpace()
    for hyper_i in hyper_space():
        print(hyper_i)

in_path="../incr_exp/uci/data/vehicle"
optim_hyper(in_path)