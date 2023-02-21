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
            ens_reader=ens.npz_writer
        self.clf_type=clf_type
        self.ens_type=ens_type
        self.ens_writer=ens_writer
        self.extractors=[]
        self.train_data=None

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

#    def 

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
        train,test= tmp_data.split()

        raise Exception((len(train),len(test)))

        return  #self.clf_alg.predict(X)
#        y=[]
#        for model_i in self.estimators_:
#            y_i=model_i.predict_proba(X)
#            y.append(y_i)
#        y=np.array(y)
#        target=np.sum(y,axis=0)
#        return np.argmax(target,axis=1)

def binarize(cat_i,targets):
    y_i=np.zeros((len(targets),2))
    for j,target_j in enumerate(targets):
        y_i[j][int(target_j==cat_i)]=1
    return y_i