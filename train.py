import sys
#from pathlib import Path
#sys.path.append(str(Path('.').absolute().parent))
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
import json
import binary,data,nn,learn,folds,utils

def multi_exp(in_path,out_path,n_iters=10,n_split=10,bayes=True):
    @utils.dir_map(depth=1)
    def helper(in_path,out_path):
        gen_data(in_path,out_path,n_iters,n_split,bayes)
    helper(in_path,out_path) 

def gen_data(in_path,out_path,n_iters=10,n_split=10,bayes=True,
                ens_type="all"):
    raw_data=data.read_data(in_path)
    data.make_dir(out_path)
    NeuralEnsemble=binary.get_ens(ens_type)
    if(bayes):
        hyperparams=find_hyperparams(raw_data,
            ensemble_type=NeuralEnsemble,n_split=n_split)
    else:
        hyperparams={'n_hidden':250,'n_epochs':200}
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
        search.fit(X_train,y_train,callback=InfoCallback(search)) 
        best_estm=search.best_estimator_
        return best_estm.get_params(deep=True)

class InfoCallback(object):
    def __init__(self,search):
        self.search=search
        self.iter=0 

    def __call__(self,optimal_result):
        self.iter+=1
#        print(optimal_result)
#        print(dir(optimal_result))
#        print(dir(self.search))
        print(f"Iteration {self.iter}")
        print(f"Best params so far {optimal_result.x}")

def find_hyperparams(train,params=None,ensemble_type=None,n_split=2):
    if(type(train)==str):
        train=data.read_data(train)
    if(params is None):
        params={'n_hidden':[25,50,100,200],
                    'n_epochs':[100,200,300,500]}
    if(ensemble_type is None):
        ensemble_type=binary.get_ens('All')
    bayes_cf=BayesOptim(ensemble_type,params,n_split=n_split)
    train_tuple=train.as_dataset()[:2]
    best_params= bayes_cf(*train_tuple)
    return best_params
    
def save_fold(ens_j,rename_j,out_j):
    data.make_dir(out_j)
    ens_j.save(f'{out_j}/models')
    with open(f'{out_j}/rename', 'wb') as f:
        json_str = json.dumps(rename_j)         
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--json", type=str, default='../uci/json/wine')
    parser.add_argument("--models", type=str, default='models')
    parser.add_argument("--bayes",action='store_true')
    parser.add_argument("--multi", type=bool, default=False)
    args = parser.parse_args()
    if(args.multi):
        fun=multi_exp
    else:
        fun=gen_data
    fun(args.json,args.models,n_iters=args.n_iters,
        n_split=args.n_split,bayes=args.bayes,ens_type="One")