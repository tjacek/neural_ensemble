import tools
tools.silence_warnings()
from time import time
from skopt.space import Real#, Categorical, Integer
import clfs,hyper

def single_exp(data_path,hyper_path,n_split,n_iter,ens_types):
    X,y=tools.get_dataset(data_path)
    bayes_optim=hyper.BayesOptim(scoring=binary_minacc,n_split=n_split,n_iter=n_iter)	
    for ens_type_i in ens_types:
#        if(clfs.is_cpu(ens_type_i)):
        print(ens_type_i)
        ens_i= clfs.get_ens(ens_type_i)
        search_i={hyper_i: Real(0.25, 5.0, prior='log-uniform') 
            for hyper_i in clfs.params_names(ens_i)}
        param_dict,best_score=bayes_optim(X,y,ens_i,search_i)

def binary_minacc(estimator, X, y):
    print(X.shape)
    raise Exception(X.shape)
    
if __name__ == '__main__':
    args=hyper.parse_args()
    if(args.clfs=='all'):
        clf_types=['CPUClf_2','CPUClf_1']
    tools.start_log(args.log_path)
    start=time()
    single_exp(args.data,args.hyper,args.n_split,
    	args.n_iter,clf_types)
    tools.log_time(f'HYPER-BINARY:{args.data}',start)     