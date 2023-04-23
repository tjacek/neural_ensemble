import argparse
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
import pandas as pd
import test,clfs
#test.silence_warnings()

class BayesOptim(object):
    def __init__(self,verbosity=True,n_iter=20,n_jobs=1):
        self.verbosity=verbosity
        self.n_iter=n_iter
        self.n_jobs=n_jobs

    def __call__(self,X,y,clf,search_spaces,n_splits):
        cv_gen=RepeatedStratifiedKFold(n_splits=n_splits, 
                    n_repeats=1, random_state=1)
        search = BayesSearchCV(estimator=clf,verbose=0,n_iter=self.n_iter,
                    search_spaces=search_spaces,n_jobs=self.n_jobs,cv=cv_gen)
        search.fit(X,y,callback=self.get_callback()) 
        return search

    def get_callback(self):
        if(self.verbosity):
            count=0
            def callback(optimal_result):
                nonlocal count
                print(f"Iteration {count}")
                print(f"Best params so far {optimal_result.x}")
                count+=1
            return callback
        return None


def single_exp(data_path,n_splits):
    df=pd.read_csv(data_path) 
    X,y=test.prepare_data(df)
    ensemble=clfs.LargeGPUClf()
    search_spaces={hyper_i:[0.5,1.0,2.0] 
            for hyper_i in ensemble.params_names()}
    bayes_optim=BayesOptim()
    bayes_optim(X,y,ensemble,search_spaces,n_splits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='csv/wine-quality-red')
    parser.add_argument("--n_splits", type=int, default=3)
    args = parser.parse_args()
    single_exp(args.data,args.n_splits)    