import tools
tools.silence_warnings()
import argparse
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.base import BaseEstimator, ClassifierMixin
import data,deep,learn,train

class ScikitAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.5, hyper=None):
        self.alpha = alpha
        self.hyper = hyper
        self.neural_ensemble=None 
        self.clfs=[]

    def fit(self,X,targets):
        ens_factory=deep.get_ensemble(('weighted',self.alpha))
        params=data.get_dataset_params(X,targets) 
        self.neural_ensemble=ens_factory(params,self.hyper)
        self.neural_ensemble.fit(X,targets)
        full=self.neural_ensemble.get_full(X) #.extract(X)
        for full_i in full:
            clf_i=learn.get_clf('RF')
            clf_i.fit(full_i,targets)
            self.clfs.append(clf_i)

    def predict_proba(self,X):
        full=self.neural_ensemble.get_full(X)
        votes=[clf_i.predict_proba(full_i) 
             for clf_i,full_i in zip(self.clfs,full)]
        votes=np.array(votes)
        return np.sum(votes,axis=0)

    def predict(self,X):
        prob=self.predict_proba(X)
        return np.argmax(prob,axis=1)

class BayesCallback(object):
    def __init__(self):
        self.count=0

    def __call__(self,optimal_result):
        print(f"Iteration {self.count}")
        print(f"Best params so far {optimal_result.x}")
        print(f'Score {round(optimal_result.fun,4)}')
        self.count+=1

def alpha_optim(data_path,hyper_path,n_split,n_repeats, n_iter):
    X,y=data.get_dataset(data_path)
    hyper_dict=train.parse_hyper(hyper_path)
    cv_gen=RepeatedStratifiedKFold(n_splits=n_split, 
                                   n_repeats=n_repeats, 
                                   random_state=1)
    search_spaces={'alpha':[0.1*(i+1) for i in range(9)]}

    search=GridSearchCV(estimator=ScikitAdapter(0.5,hyper_dict),
                        param_grid=search_spaces,
                        verbose=1)

#    search_spaces={'alpha': Real(0.1, 0.9, prior='uniform')}
#    search = BayesSearchCV(estimator=ScikitAdapter(0.5,hyper_dict),
#                           n_iter=n_iter,
#                           search_spaces=search_spaces,
#                           cv=cv_gen,
#                           scoring='accuracy',
#                           verbose=0,
#                           n_jobs=1,)
    search.fit(X,y) #,callback=BayesCallback()) 
    print(search.cv_results_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../data/cleveland')
    parser.add_argument("--hyper", type=str, default='../test3/hyper/cleveland')
    parser.add_argument("--n_split", type=int, default=3)
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--log", type=str, default='log')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
#    tools.start_log(args.log)
    alpha_optim(args.data,args.hyper,args.n_split,args.n_iter,5)