from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import combinations
from keras import callbacks
import tensorflow as tf
import numpy as np
import data,nn,learn

class NeuralEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self,n_hidden=250,multi_clf='LR'):#n_epochs=200):
        self.n_hidden=n_hidden
        self.multi_clf=multi_clf
#        self.n_epochs=n_epochs

    def fit(self,X,targets):
        raise NotImplementedError

    def predict_proba(self,X):
        votes=[]
        for extr_i,model_i in zip(self.extractors,self.models):
            binary_i=extr_i.predict(X)
            full_i=np.concatenate([X,binary_i],axis=1)
            vote_i= model_i.predict_proba(full_i)
            votes.append(vote_i)
        votes=np.array(votes)
        prob=np.sum(votes,axis=0)
        return prob

    def predict(self,X):
        prob=self.predict_proba(X)
        return np.argmax(prob,axis=1)

    def save(self,out_path):
        data.make_dir(out_path)
        for i,extr_i in enumerate(self.extractors):
            extr_i.save(f'{out_path}/{i}')

    def train_extractor(self,X,y_i):
        earlystopping = callbacks.EarlyStopping(monitor="accuracy",
                mode="min", patience=5,restore_best_weights=True)
        nn_params={'dims':X.shape[1],'n_cats':2}
        nn_i=nn.SimpleNN(n_hidden=int(self.n_hidden))(nn_params)
        nn_i.fit(X,y_i,epochs=500,#self.n_epochs,
            batch_size=32,verbose = 0,callbacks=earlystopping)
        extractor_i= nn.get_extractor(nn_i)
        self.extractors.append(extractor_i)
        return extractor_i

    def train_model(self,X,binary_i,targets):
        clf_i=learn.get_clf(self.multi_clf)#'LR')
        full_i=np.concatenate([X,binary_i],axis=1)
        clf_i.fit(full_i,targets)#,callbacks=earlystopping)
        self.models.append(clf_i)

class OneVsAll(NeuralEnsemble):
    def fit(self,X,targets):
        self.extractors=[]
        self.models=[]
        n_cats=max(targets)+1
        for cat_i in range(n_cats):
            y_i=binarize(cat_i,targets)
            extractor_i=self.train_extractor(X,y_i)
            binary_i=extractor_i.predict(X)
            self.train_model(X,binary_i,targets)

class OneVsOne(NeuralEnsemble):
    def fit(self,X,targets):
        self.extractors=[]
        self.models=[]
        n_cats=max(targets)+1
        pairs = list(combinations(range(n_cats),2))
        for i,j in pairs:
            selected=[k for k,y_k in enumerate(targets)
                        if((y_k==i) or (y_k==j))]
            y_s=[ int(targets[k]==i) for k in selected]                
            y_s= tf.keras.utils.to_categorical(y_s, num_classes = 2)
            X_s=np.array([ X[k,:] for k in selected])
            extractor_i=self.train_extractor(X_s,y_s)
            binary_i=extractor_i.predict(X)
            self.train_model(X,binary_i,targets)

def binarize(cat_i,targets):
    y_i=np.zeros((len(targets),2))
    for j,target_j in enumerate(targets):
        y_i[j][int(target_j==cat_i)]=1
    return y_i

def get_ens(ens_type):
    if(ens_type=='One'):
        return OneVsOne
    return OneVsAll