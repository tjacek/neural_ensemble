class Experiment(object):
    def __init__(self,split,params,hyper_params=None,model=None):
        self.split=split
        self.params=params
        self.hyper_params=hyper_params
        self.model=model