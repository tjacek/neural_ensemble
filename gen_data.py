from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
import data,binary

class Protocol(object):
    def __init__(self,search_space=None):
        if(search_space is None):
            search_space={'n_hidden':[25,50,100,200],
                          'n_epochs':[100,250,500]}
        self.search_space=search_space

    def __call__(self,in_path,n_split=10):
        ensemble_type=binary.SciktFacade
        hyperparams=find_hyperparams(in_path,self.search_space,
            ensemble_type=ensemble_type,n_split=n_split)
        print(hyperparams)

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

in_path='../imb_json/cleveland'
protocol=Protocol()
protocol(in_path)