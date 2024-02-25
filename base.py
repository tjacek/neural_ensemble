import json,random


class Experiment(object):
    def __init__(self,split,params,hyper_params=None,model=None):
        self.split=split
        self.params=params
        self.hyper_params=hyper_params
        self.model=model

class AlgParams(object):
    def __init__(self,hyper_type='eff',epochs=300,callbacks=None,alpha=None,
                    bayes_iter=5,rest_clf=None):
        if(alpha is None):
            alpha=[0.25,0.5,0.75]
        self.hyper_type=hyper_type
        self.epochs=epochs
        self.alpha=alpha
        self.bayes_iter=bayes_iter
        self.rest_clf=rest_clf

    def optim_alpha(self):
        return type(self.alpha)==list 

    def get_callback(self):
        return tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=5)

class Protocol(object):
    def __init__(self,n_split=10,n_iters=10):
        self.n_split=n_split
        self.n_iters=n_iters
        self.current_split=None

    def set_split(self,dataset):
        return

class Split(object):
    def __init__(self,dataset,y,train,valid,test):
        self.dataset=dataset
        self.train=train
#        self.valid=valid
        self.test=test
    
    def get_train(self):
        return self.dataset.X[self.train],self.dataset.y[self.train]
    
#    def get_valid(self):
#        return self.dataset.X[self.valid],self.dataset.y[self.valid]

    def get_test(self):
        return self.dataset.X[self.test],self.dataset.y[self.test]

def split_data(X,y):
    by_cat=defaultdict(lambda :[])
    for i,cat_i in enumerate( y):
        by_cat[cat_i].append(i)
    train,valid,test=[],[],[]
    for cat_i,samples_i in by_cat.items():
        random.shuffle(samples_i)
        for j,index in enumerate(samples_i):
            mod_j= j % 5
            if(mod_j==4):
                valid.append(index)
            elif(mod_j==0):
                test.append(index)
            else:
                train.append(index)
    return train,valid,test   