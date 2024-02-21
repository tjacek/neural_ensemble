class Experiment(object):
    def __init__(self,split,params,hyper_params=None,model=None):
        self.split=split
        self.params=params
        self.hyper_params=hyper_params
        self.model=model

class Protocol(object):
    def __init__(self,n_split=10,n_iters=10):
        self.n_split=n_split
        self.n_iters=n_iters