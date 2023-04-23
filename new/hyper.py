import argparse
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
import pandas as pd
import test,clfs

class BayesOptim(object):
    def __init__(self,verbosity=True,n_iter=5,n_split=3):
        self.verbosity=verbosity
        self.n_iter=n_iter
        self.n_split=n_split
    
    def __call__(self,X,y,clf,search_spaces):
        cv_gen=RepeatedStratifiedKFold(n_splits=self.n_split, 
                    n_repeats=1, random_state=1)
        search = BayesSearchCV(estimator=clf,verbose=0,n_iter=self.n_iter,
                    search_spaces=search_spaces,n_jobs=1,cv=cv_gen)
        callback=BayesCallback() if(self.verbosity) else None
        search.fit(X,y,callback=callback) 
        best_estm=search.best_estimator_
        return best_estm.get_params(deep=True)

    def get_setting(self):
        return f'bayes_iter:{self.n_iter},n_split:{self.n_split}' 

class BayesCallback(object):
    def __init__(self):
        self.count=0

    def __call__(self,optimal_result):
        print(f"Iteration {self.count}")
        print(f"Best params so far {optimal_result.x}")
        print(f'Score {optimal_result.fun}')
        self.count+=1

def single_exp(data_path,hyper_path,n_split,n_iter,ens_types):
    df=pd.read_csv(data_path) 
    X,y=test.prepare_data(df)
    bayes_optim=BayesOptim( n_split=n_split,n_iter=n_iter)
#    ens_types=[clfs.GPUClf_2_2() ,clfs.CPUClf_2()]
    with open(hyper_path,"a") as f:
        f.write(f'data:{data_path},{bayes_optim.get_setting()}\n')
        for ens_type_i in ens_types:
            ens_i= clfs.get_ens(ens_type_i)
            search_i={hyper_i:[0.5,1.0,2.0] 
                for hyper_i in ens_i.params_names()}
            if(clfs.is_cpu(ens_i)):
                search_i['multi_clf']=['RF']	
            param_dict=bayes_optim(X,y,ens_i,search_i)
            print(param_dict)
            ens_name=ens_i.__class__.__name__
            f.write(f'{ens_name},{str(param_dict)}\n') 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='csv/wine-quality-red')
    parser.add_argument("--hyper", type=str, default='hyper.txt')
    parser.add_argument("--n_split", type=int, default=3)
    parser.add_argument("--n_iter", type=int, default=5)

    clf_types=clfs.CLFS_NAMES
    args = parser.parse_args()
    single_exp(args.data,args.hyper,args.n_split,
    	args.n_iter,clf_types)    