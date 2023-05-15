import tools
tools.silence_warnings()
import argparse
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from time import time
import pandas as pd
import test,clfs,tools

class BayesOptim(object):
    def __init__(self,verbosity=True,n_iter=5,n_split=3):
        self.verbosity=verbosity
        self.n_iter=n_iter
        self.n_split=n_split
    
    def __call__(self,X,y,clf,search_spaces):
        cv_gen=RepeatedStratifiedKFold(n_splits=self.n_split, 
                    n_repeats=3, random_state=1)
        search = BayesSearchCV(estimator=clf,verbose=0,n_iter=self.n_iter,
                search_spaces=search_spaces,n_jobs=1,cv=cv_gen,
                scoring='balanced_accuracy')
        callback=BayesCallback() if(self.verbosity) else None
        search.fit(X,y,callback=callback) 
        best_estm=search.best_estimator_
        best_score=round(search.best_score_,4)
        return best_estm.get_params(deep=True),best_score

    def get_setting(self):
        return f'bayes_iter:{self.n_iter},n_split:{self.n_split}' 

class BayesCallback(object):
    def __init__(self):
        self.count=0

    def __call__(self,optimal_result):
        print(f"Iteration {self.count}")
        print(f"Best params so far {optimal_result.x}")
        print(f'Score {round(optimal_result.fun,4)}')
        self.count+=1

def single_exp(data_path,hyper_path,n_split,n_iter,ens_types):
    df=pd.read_csv(data_path) 
    X,y=tools.prepare_data(df)
    bayes_optim=BayesOptim( n_split=n_split,n_iter=n_iter)
    with open(hyper_path,"a") as f:
        f.write(f'data:{data_path},{bayes_optim.get_setting()}\n')

    for ens_type_i in ens_types:
        ens_i= clfs.get_ens(ens_type_i)
        search_i={hyper_i: Real(0.25, 5.0, prior='log-uniform') 
                for hyper_i in clfs.params_names(ens_i)}
        if(clfs.is_cpu(ens_i)):
            search_i['multi_clf']=['RF']	
        param_dict,best_score=bayes_optim(X,y,ens_i,search_i)
        param_dict={ key_i:round_hype(hyper_i) 
            for key_i,hyper_i in param_dict.items()}
        print(param_dict)
        ens_name=ens_i.__class__.__name__
        with open(hyper_path,"a") as f:
            f.write(f'{ens_name},{str(param_dict)},{best_score}\n') 

def round_hype(hyper_i):
    if(type(hyper_i)==float):
        return round(hyper_i,4)
    return hyper_i

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='uci/newthyroid')
    parser.add_argument("--hyper", type=str, default='newthyroid/hyper.txt')
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--clfs", type=str, default='all')
    parser.add_argument("--log_path", type=str, default='newthyroid/log.time')

    args = parser.parse_args()
#    print("@@@@@@@@@@@@@@@@@@")
#    print(args.clfs)
#    raise Exception(args.clfs)

    if(args.clfs=='all'):
        clf_types=clfs.CLFS_NAMES
    else:
    	clf_types= args.clfs.split(',')

    tools.start_log(args.log_path)
    start=time()
    single_exp(args.data,args.hyper,args.n_split,
    	args.n_iter,clf_types)
    tools.log_time(f'HYPER:{args.data}',start) 