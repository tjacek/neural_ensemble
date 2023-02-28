import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import json
import data,nn,learn,folds,utils

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
        return np.argmax(prob,axis=1)

    def save(self,out_path):
        data.make_dir(out_path)
        for i,extr_i in enumerate(self.extractors):
            extr_i.save(f'{out_path}/{i}')

class BayesOptim(object):
    def __init__(self,clf_alg,search_spaces,n_split=5):
        self.clf_alg=clf_alg 
        self.n_split=n_split
        self.search_spaces=search_spaces

    def __call__(self,X_train,y_train):
        cv_gen=RepeatedStratifiedKFold(n_splits=self.n_split, 
                n_repeats=1, random_state=1)
        search = BayesSearchCV(estimator=self.clf_alg(), 
            search_spaces=self.search_spaces,n_jobs=-1,cv=cv_gen)
        search.fit(X_train,y_train) 
        best_estm=search.best_estimator_
        return best_estm.get_params(deep=True)

def find_hyperparams(train,params=None,ensemble_type=None,n_split=2):
    if(type(train)==str):
        train=data.read_data(train)
    if(params is None):
        params={'n_hidden':[25,50,100,200],
                    'n_epochs':[100,200,300,500]}
    if(ensemble_type is None):
        ensemble_type=NeuralEnsemble
    bayes_cf=BayesOptim(ensemble_type,params,n_split=n_split)
    train_tuple=train.as_dataset()[:2]
    best_params= bayes_cf(*train_tuple)
    return best_params

def gen_data(in_path,out_path,n_iters=10,n_split=10,bayes=True):
    raw_data=data.read_data(in_path)
    data.make_dir(out_path)
    if(bayes):
        hyperparams=find_hyperparams(raw_data)
    else:
        hyperparams={'n_hidden':200,'n_epochs':200}
    for i in range(n_iters):
        out_i=f'{out_path}/{i}'
        data.make_dir(out_i)
        folds_i=folds.make_folds(raw_data,k_folds=n_split)
        splits_i=folds.get_splits(raw_data,folds_i)
        for j,(data_j,rename_j) in enumerate(splits_i):
            ens_j= NeuralEnsemble(**hyperparams)
            learn.fit_clf(data_j,ens_j)
            out_j=f'{out_i}/{j}'
            save_fold(ens_j,rename_j,out_j)
    
def binarize(cat_i,targets):
    y_i=np.zeros((len(targets),2))
    for j,target_j in enumerate(targets):
        y_i[j][int(target_j==cat_i)]=1
    return y_i

def save_fold(ens_j,rename_j,out_j):
    data.make_dir(out_j)
    ens_j.save(f'{out_j}/models')
    with open(f'{out_j}/rename', 'wb') as f:
        json_str = json.dumps(rename_j)         
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)

def multi_exp(in_path,out_path,n_iters=10,n_split=10,bayes=True):
    @utils.dir_map(depth=1)
    def helper(in_path,out_path):
        gen_data(in_path,out_path,n_iters,n_split,bayes)
    helper(in_path,out_path) 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--json", type=str, default='../bayes/json/')
    parser.add_argument("--models", type=str, default='../bayes/models')
    parser.add_argument("--bayes", type=bool, default=True)
    parser.add_argument("--multi", type=bool, default=True)
    args = parser.parse_args()
    if(args.multi):
        fun=multi_exp
    else:
        fun=gen_data
    fun(args.json,args.models,n_iters=args.n_iters,
        n_split=args.n_split,bayes=args.bayes)