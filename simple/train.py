import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.linear_model import LogisticRegression
import data,nn,learn

class NeuralEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self,n_hidden=200,n_epochs=200):
        self.n_hidden=n_hidden
        self.n_epochs=n_epochs

    def fit(self,X,targets):
        self.extractors=[]
        self.models=[]
        n_cats=max(targets)+1
        for cat_i in range(n_cats):
            y_i=binarize(cat_i,targets)
            nn_params={'dims':X.shape[1],'n_cats':2}
            nn_i=nn.SimpleNN(n_hidden=self.n_hidden)(nn_params)
            nn_i.fit(X,y_i,
                epochs=self.n_epochs,batch_size=32)
            extractor_i= nn.get_extractor(nn_i)
            self.extractors.append(extractor_i)
            binary_i=extractor_i.predict(X)
            clf_i=learn.get_clf('LR')
            full_i=np.concatenate([X,binary_i],axis=1)
            clf_i.fit(full_i,targets)
            self.models.append(clf_i)

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
        return np.argmax(prob,axis=0)
        
def experiment(in_path):
    raw_data=data.read_data(in_path)
    ne= NeuralEnsemble()
    result=learn.fit_clf(raw_data,clf_type=ne)
    print(result.get_acc())
    print(dir(ne))
    
def binarize(cat_i,targets):
    y_i=np.zeros((len(targets),2))
    for j,target_j in enumerate(targets):
        y_i[j][int(target_j==cat_i)]=1
    return y_i

in_path='../../uci/json/wine'
experiment(in_path)