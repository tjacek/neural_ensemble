import tools
tools.silence_warnings()
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import Input, Model
import keras_tuner as kt
from sklearn.base import BaseEstimator, ClassifierMixin
import argparse
import data,deep,learn

class ScikitAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha, hyper):
        self.alpha = alpha
        self.hyper = hyper
        self.neural_ensemble=None 
        self.clfs=[]

    def fit(self,X,targets):
        ens_factory=deep.get_ensemble('weighted')
        params=get_dataset_params(X) 
        self.neural_ensemble=ens_factory(params,self.hyper)
        self.neural_ensemble.fit(self,X,targets)
        full=self.neural_ensemble.get_full(X) #.extract(X)
        for full_i in full:
            clf_i=learn.get_clf('RF')
            clf_i.fit(full_i,targets)
            self.clfs.append(clf_i)

    def predict_proba(self,X):    
        votes=[clf_i.predict_proba(X) 
             for clf_i in self.clfs]
        votes=np.array(votes)
        return np.sum(votes,axis=0)

    def predict(self,X):
        prob=self.predict_proba(X)
        return np.argmax(prob,axis=1)

class MultiKTBuilder(object): 
    def __init__(self,params):#,hidden=[(0.25,5),(0.25,5)]):
        self.params=params
        self.hidden=(1,10)

    def __call__(self,hp):
        model = tf.keras.Sequential()
        for i in range(hp.Int('layers', 1, 2)):
            hidden_i=hp.Int('units_' + str(i), 
                    min_value= int(self.params['dims']*self.hidden[0]), 
                    max_value=int(self.params['dims']*self.hidden[1]),
                    step=10)
            model.add(tf.keras.layers.Dense(units=hidden_i ))
        batch=hp.Choice('batch', [True, False])
        if(batch):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(self.params['n_cats'], activation='softmax'))
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

@tools.log_time(task='HYPER')
def single_exp(data_path,hyper_path,n_split,n_iter):
    X,y=data.get_dataset(data_path)
    data_params=data.get_dataset_params(X,y)
    print(data_params)
    best=bayes_optim(X,y,data_params,n_split,n_iter)
    best=[tools.round_data(best_i,4) for best_i in best]
    with open(hyper_path,"a") as f:
        f.write(f'{str(best)}\n') 
    return best

def bayes_optim(X,y,data_params,n_split,n_iter):
    model_builder= MultiKTBuilder(data_params) 

    tuner=kt.BayesianOptimization(model_builder,
                objective='loss',#'val_loss',
                max_trials=n_iter,
                overwrite=True)
    validation_split= 1.0/n_split
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
        patience=50)
    tuner.search(X, y, epochs=150, validation_split=validation_split,
       verbose=1,callbacks=[stop_early])
    
    tuner.results_summary()
    best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
    best=  best_hps.values
    relative={key_i: (value_i/data_params['dims'])  
                for key_i,value_i in best.items()
                    if('unit' in key_i)}
    models=tuner.get_best_models()
    acc=get_metric_value(tuner,X,y)
    return best,relative,acc

def get_metric_value(tuner,X,y):
    best_model= tuner.get_best_models(1)[0]
    metric_values= best_model.evaluate(X, y)
    eval_metrics= list(zip(best_model.metrics_names ,metric_values))
    acc=[]
    for metric_i,value_i in eval_metrics:
        if('accuracy' in metric_i):
            acc.append(value_i)
    return acc[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='data')# /wine-quality-red')
    parser.add_argument("--hyper", type=str, default='hyper')
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--log", type=str, default='log')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    tools.start_log(args.log)
    if(args.dir>0):
        single_exp=tools.dir_fun(2)(single_exp)
    single_exp(args.data,args.hyper,args.n_split,args.n_iter)