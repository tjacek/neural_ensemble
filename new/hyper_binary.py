import tools
tools.silence_warnings()
from time import time
import numpy as np
from skopt.space import Real#, Categorical, Integer
import clfs,hyper

def single_exp(data_path,hyper_path,n_split,n_iter,ens_types):
    X,y=tools.get_dataset(data_path)
    bayes_optim=hyper.BayesOptim(scoring=max_acc,n_split=n_split,
        n_repeats=1,n_iter=n_iter)	
    with open(hyper_path,"a") as f:
        f.write(f'data:{data_path},{bayes_optim.get_setting()}\n')
    for ens_type_i in ens_types:
#        if(clfs.is_cpu(ens_type_i)):
        print(ens_type_i)
        ens_i= clfs.get_ens(ens_type_i)
        search_i={hyper_i: Real(0.25, 7.0, prior='uniform') #'log-uniform') 
            for hyper_i in clfs.params_names(ens_i)}
        param_dict,best_score=bayes_optim(X,y,ens_i,search_i)
        param_dict={ key_i:hyper.round_hype(hyper_i) 
            for key_i,hyper_i in param_dict.items()}
        print(param_dict)
        ens_name=ens_i.__class__.__name__
        with open(hyper_path,"a") as f:
            f.write(f'{ens_name},{str(param_dict)},{best_score}\n')

def min_acc(estimator, X, y):
    acc_desc=estimator.catch
    acc_min= min(list(acc_desc.values()))  
    return acc_min

def max_acc(estimator, X, y):
    acc_desc=estimator.catch
    acc_min= min(list(acc_desc.values()))  
    return acc_min

def mean_acc(estimator, X, y):
    acc_desc=estimator.catch
    acc_mean= np.mean(list(acc_desc.values()))  
    return acc_mean


if __name__ == '__main__':
    args=hyper.parse_args()
    if(args.clfs=='all'):
        clf_types=['CPUClf_2','CPUClf_1']
    tools.start_log(args.log_path)
    start=time()
    single_exp(args.data,args.hyper,args.n_split,
    	args.n_iter,clf_types)
    tools.log_time(f'HYPER-BINARY:{args.data}',start)     