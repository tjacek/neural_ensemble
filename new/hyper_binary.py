import tools
tools.silence_warnings()
from time import time
import numpy as np
from skopt.space import Real#, Categorical, Integer
from sklearn.metrics import accuracy_score
import clfs,hyper,learn

def single_exp(data_path,hyper_path,n_split,n_iter,ens_types):
    X,y=tools.get_dataset(data_path)
    bayes_optim=hyper.BayesOptim(scoring=multi_acc,n_split=n_split,
        n_repeats=1,n_iter=n_iter)	
    with open(hyper_path,"a") as f:
        f.write(f'data:{data_path},{bayes_optim.get_setting()}\n')
    for ens_type_i in ens_types:
#        if(clfs.is_cpu(ens_type_i)):
        print(ens_type_i)
        ens_i= clfs.get_ens(ens_type_i)
        search_i=get_search_space(ens_i)
        param_dict,best_score=bayes_optim(X,y,ens_i,search_i)
        param_dict={ key_i:hyper.round_hype(hyper_i) 
            for key_i,hyper_i in param_dict.items()}
        print(param_dict)
        ens_name=ens_i.__class__.__name__
        with open(hyper_path,"a") as f:
            f.write(f'{ens_name},{str(param_dict)},{best_score}\n')

def get_search_space(ens_i):
    search_i={hyper_i: Real(1.0, 7.0, prior='uniform') #'log-uniform') 
            for hyper_i in clfs.params_names(ens_i)}
    search_i['multi_clf']=['RF','SVC']
    return search_i

def multi_acc(estimator, X_test, y_test):
    X_train,y_train= estimator.train_data
    clf_type= estimator.multi_clf
    binary_train=estimator.binary_model.predict(X_train)
    clfs=[]
    for binary_i in binary_train:
        clf_i =learn.get_clf(clf_type)
        clf_i.fit(binary_i,y_train)
        clfs.append(clf_i)

    binary_test=estimator.binary_model.predict(X_test)
    acc=[]
    for i,binary_i in enumerate( binary_test):
        pred_i=clfs[i].predict(binary_i)
        acc_i= accuracy_score( y_test,pred_i)
        acc.append(acc_i)
    return np.mean(acc)

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