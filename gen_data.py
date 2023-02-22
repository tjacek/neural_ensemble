from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
import data,binary

class Protocol(object):
    def __init__(self,search_space=None):
        if(search_space is None):
            search_space={'n_hidden':[25,50,100,200],
                          'n_epochs':[100,250,500]}
        self.search_space=search_space

    def __call__(self,in_path,n_iters=10,n_split=10):
        ensemble_type=binary.SciktFacade
#        hyperparams=find_hyperparams(in_path,self.search_space,
#            ensemble_type=ensemble_type,n_split=n_split)
        raw_data=data.read_data(in_path)
        for i in range(n_iters):
            folds_i=make_folds(raw_data,k_folds=n_split)
            for data_j in get_splits(raw_data,folds_i):
                raise Exception(data_j.keys())

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
        search.fit(X_train,y_train) 
        best_estm=search.best_estimator_
        return best_estm.get_params(deep=True)

def find_hyperparams(train,params,ensemble_type=None,n_split=2):
    if(type(train)==str):
        train=data.read_data(train)
    bayes_cf=BayesOptim(ensemble_type,params,n_split=n_split)
    train_tuple=train.as_dataset()[:2]
    best_params= bayes_cf(*train_tuple)
    return best_params

def make_folds(data_dict,k_folds=10):
    if(type(data_dict)==str):
        data_dict=data.read_data(data_dict)
    names=data_dict.names()
    folds=[[] for i in range(k_folds)]
    cats=names.by_cat()
    for cat_i in cats.values():
        cat_i.shuffle()
        for j,name_j in enumerate(cat_i):
            folds[j % k_folds].append(name_j)
    return folds

def get_splits(data_dict,folds):
    for i in range(len(folds)):
        test=folds[i]
        train=[]
        for j,fold_j in enumerate(folds):
            if(i!=j):
                train+=fold_j
        new_names={}
        for name_i in train:
            name_i=data.Name(name_i)
            new_names[name_i]=name_i.set_train(False)
        for name_i in test:
            name_i=data.Name(name_i)
            new_names[name_i]=name_i.set_train(True)
        yield data_dict.rename(new_names)

in_path='wine.json'#'../imb_json/cleveland'
protocol=Protocol()
protocol(in_path)