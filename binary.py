from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import numpy as np
import ens,nn,data

class NECSCF(object):
    def __init__(self,clf_type='LR',ens_type=None,
            ens_writer=None):
        if(ens_type is None):
            ens_type=ens.Ensemble
        if(ens_writer is None):
            ens_writer=ens.npz_writer
        self.clf_type=clf_type
        self.ens_type=ens_type
        self.ens_writer=ens_writer
        self.extractors=[]
        self.train_data=None

    def  __call__(self,common):
        binary=self.gen_binary(common)
        ens_instance=self.ens_type(common,binary)
        return ens_instance#.evaluate()

    def fit(self,data,hyper):
        train=data.split()[0]
        X_train,y_train,names=train.as_dataset()
        n_hidden,n_epochs=hyper['n_hidden'],hyper['n_epochs']
        batch_size=hyper['batch_size']
        self.make_extractor(X_train,y_train,
            n_hidden,n_epochs,batch_size)

    def make_extractor(self,X,targets,n_hidden=200,
            n_epochs=200,batch_size=32):
        n_cats=max(targets)+1
        for cat_i in range(n_cats):
            y_i=binarize(cat_i,targets)
            nn_params={'dims':X.shape[1],'n_cats':2}
            model_i=nn.SimpleNN(n_hidden=n_hidden)(nn_params)
            model_i.fit(X,y_i,
                epochs=n_epochs,batch_size=batch_size)
            extractor_i= nn.get_extractor(model_i)
            self.extractors.append(extractor_i)
        self.train_data=(X,targets)
        return self.extractors

    def gen_binary(self,common):
        binary=[]
        X,y,names=common.as_dataset()
        for extractor_i in self.extractors:
            binary_i=extractor_i.predict(X)
            pairs_i=[ (name_j,x_ij) 
                for name_j,x_ij in zip(names,binary_i)]
            binary.append(data.DataDict(pairs_i))
        return binary

class SciktFacade(BaseEstimator, ClassifierMixin):
    def __init__(self,n_hidden=200,n_epochs=200,necscf=None):
        if(necscf is None):
            necscf=NECSCF()
        self.necscf=necscf
        self.n_hidden=n_hidden
        self.n_epochs=n_epochs
        self.batch_size=32

    def fit(self,X,targets):
        self.necscf.make_extractor(X,targets,self.n_hidden,
            self.n_epochs,self.batch_size)
        return self     

    def predict(self,X):
        X_train,y_train= self.necscf.train_data
        tmp_data=data.from_tuple(X_train,y_train,test=False)
        for i,x_i in enumerate(X):
            name_i=data.Name(f'1_1_{i}')
            tmp_data[name_i]=x_i
        
        ens_inst= self.necscf(tmp_data)
        result=ens_inst.evaluate()
        y_pred=  result.get_pred()[0]
        return y_pred

def binarize(cat_i,targets):
    y_i=np.zeros((len(targets),2))
    for j,target_j in enumerate(targets):
        y_i[j][int(target_j==cat_i)]=1
    return y_i