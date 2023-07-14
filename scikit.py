import tools
tools.silence_warnings()
import argparse
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.base import BaseEstimator, ClassifierMixin
import data,deep,learn

class ScikitAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha, hyper):
        self.alpha = alpha
        self.hyper = hyper
        self.neural_ensemble=None 
        self.clfs=[]

    def fit(self,X,targets):
        ens_factory=deep.get_ensemble('weighted')
        params=get_dataset_params(X) 
        self.neural_ensemble=ens_factory(params,self.hyper)
        self.neural_ensemble.fit(self,X,targets)
        full=self.neural_ensemble.get_full(X) #.extract(X)
        for full_i in full:
            clf_i=learn.get_clf('RF')
            clf_i.fit(full_i,targets)
            self.clfs.append(clf_i)

    def predict_proba(self,X):    
        votes=[clf_i.predict_proba(X) 
             for clf_i in self.clfs]
        votes=np.array(votes)
        return np.sum(votes,axis=0)

    def predict(self,X):
        prob=self.predict_proba(X)
        return np.argmax(prob,axis=1)

def alpha_optim(hyper,n_split,n_repeats, n_iter):
    search_space={'alpha': Real(0.1, 0.9, prior='uniform'),
                  'hyper':[hyper] }
    cv_gen=RepeatedStratifiedKFold(n_splits=n_split, 
                n_repeats=n_repeats, random_state=1)
    search = BayesSearchCV(estimator=clf,verbose=0,n_iter=self.n_iter,
                search_spaces=search_spaces,n_jobs=1,cv=cv_gen,
                scoring=self.scoring)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='data')# /wine-quality-red')
    parser.add_argument("--hyper", type=str, default='hyper')
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--log", type=str, default='log')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    tools.start_log(args.log)