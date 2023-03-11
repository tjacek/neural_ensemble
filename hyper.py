from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import binary, conf,data

class HyperOptimisation(object):
    def __init__(self, search_alg,search_spaces=None):
#        if(default_params is None):
#            default_params={'n_hidden':250,'n_epochs':200}
        if(search_spaces is None):
            search_spaces={'n_hidden':[25,50,100,200],
                    'n_epochs':[100,200,300,500]}
#        self.default_params=default_params
        self.search_spaces=search_spaces
        self.search_alg= search_alg #GridOptim() 

    def params_name(self):
        return self.search_spaces.keys()
#    def __call__(self,train,ensemble=None,n_split=10):
#        if(self.search_spaces):
#            print('Optimisation of hyperparams')
#            return self.optim(train,ensemble,n_split)
#        else:
#            return self.default_params

    def __call__(self,train,ensemble=None,n_split=10):
        if(type(train)==str):
            train=data.read_data(train)
        ensemble=binary.get_ens(ensemble)
        def helper(X_train,y_train):
            cv_gen=RepeatedStratifiedKFold(n_splits=n_split, 
                    n_repeats=1, random_state=1)
            search=self.search_alg((X_train,y_train),ensemble(),
                self.search_spaces,cv_gen)
            best_estm=search.best_estimator_
            return best_estm.get_params(deep=True)
        train_tuple=train.as_dataset()[:2]
        best_params= helper(*train_tuple)
        return best_params

class BayesOptim(object):
    def __init__(self,n_jobs=1,verbosity=True,n_iter=20):
        self.n_jobs=n_jobs
        self.verbosity=verbosity
        self.n_iter=n_iter

    def __call__(self,train_tuple,clf,search_spaces,cv_gen):
        search = BayesSearchCV(estimator=clf,verbose=0,n_iter=self.n_iter,
                    search_spaces=search_spaces,n_jobs=self.n_jobs,cv=cv_gen)
        X_train,y_train=train_tuple
        search.fit(X_train,y_train,callback=self.get_callback()) 
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

class GridOptim(object):
    def __init__(self,n_jobs=1):
        self.n_jobs=n_jobs

    def __call__(self,train_tuple,clf,search_spaces,cv_gen):
        search = GridSearchCV(estimator=clf,param_grid=search_spaces,
#                  verbose=0,
                  n_jobs=self.n_jobs,cv=cv_gen)
        X_train,y_train=train_tuple
        search.fit(X_train, y_train)
        return search

def hyper_exp(conf_path,n_split):
    dir_dict,hyper_dict=conf.read_hyper(conf_path)
    print(hyper_dict)
    hyper_optim=parse_hyper(hyper_dict)
    optimal_result={}
    for path_i in data.top_files(dir_dict['json']):
        raw_data=data.read_data(path_i)
        hyperparams=hyper_optim(raw_data,"all",n_split)
        optimal_result[path_i.split('/')[-1]]=hyperparams
    print(optimal_result) 

def parse_hyper(conf_dict):
    if(conf_dict['optim_type']=='grid'):
        search_alg= GridOptim(int(conf_dict['n_jobs']))
    if(conf_dict['optim_type']=='bayes'):
        n_jobs,n_iter= int(conf_dict['n_jobs']),int(conf_dict['bayes_iter'])
        verbosity=conf_dict['verbosity']
        search_alg= BayesOptim(n_jobs,verbosity,n_iter)
    return HyperOptimisation(search_alg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_split", type=int, default=3)
    parser.add_argument("--conf",type=str,default='conf/base.cfg')
    args = parser.parse_args()
#    conf_dict=conf.read_conf(args.conf,'HYPER')
    hyper_exp(args.conf,args.n_split)