import utils
utils.silence_warnings()
import numpy as np
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import Input, Model
import keras_tuner as kt
from sklearn.utils import class_weight
import argparse,json
import base,deep,data,protocol

class MultiKTBuilder(object): 
    def __init__(self,params):
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

    def extract_hyper(self,tuner):
        best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
        best=  best_hps.values
        return best

class EffBuilder(object):
    def __init__(self,params):
        self.params=params
        self.first=(1,10)
        self.second=(3,20)

    def __call__(self,hp):
        model = tf.keras.Sequential()
        dims=int(self.params['dims'])
        first_hp=hp.Int('units_0', 
                        min_value= dims*self.first[0], 
                        max_value=dims*self.first[1],
                        step=10)
        model.add(tf.keras.layers.Dense(units=first_hp))
        n_cats=int(self.params['n_cats'])
        second_hp=hp.Int('units_1', 
                          min_value=n_cats*self.second[0], 
                          max_value=n_cats*self.second[1],
                          step=10)
        model.add(tf.keras.layers.Dense(units=second_hp))
        batch=hp.Choice('batch', [True, False])
        if(batch):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(self.params['n_cats'], 
                                        activation='softmax'))
        model.compile('adam', 
                      'sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def extract_hyper(self,tuner):
        best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
        best=  best_hps.values
        best['layers']=2
        return best

def bayes_optim(dataset,alg_params,protocol,verbose=1):
    model_builder=get_builder(alg_params,dataset.params)
    tuner=kt.BayesianOptimization(model_builder,
                objective='val_loss',
                max_trials= alg_params.bayes_iter,
                overwrite=True)
    split=protocol.split_gen.single_split(dataset)
    x_train,y_train=split.get_train()
    x_valid,y_valid=split.get_valid()
#    class_weights = class_weight.compute_class_weight(class_weight='balanced',
#                                                      classes=np.unique(y_train),
#                                                      y=y_train)
    tuner.search(x=x_train, 
                 y=y_train, 
                 epochs=150,
                 batch_size=dataset.params['batch'], 
                 validation_data=(x_valid, y_valid),
                 verbose=verbose,
                 callbacks=[alg_params.get_callback()])
#[alg_params.callbacks])#,
#                 class_weight=exp.params['class_weights'])
    
    tuner.results_summary()
    return model_builder.extract_hyper(tuner)

def get_builder(alg_params,params):
    if(alg_params.hyper_type=='eff'):
        return EffBuilder(params)
    if(alg_params.hyper_type=='multi'):
        return MultiKTBuilder(params)
    return alg_params.hyper_type(params)

def find_alpha(alg_params,split,params,hyper_dict,verbose=1):
    alpha=[0.25,0.5,0.75]
    acc,all_exp=[],[]
    for alpha_i in alpha:
        exp_i=base.Experiment(split=split,
                              params=params,
                              hyper_params=hyper_dict,
                              model=None)
        exp_i.train(verbose=0,
                    alg_params=alg_params,
                    alpha=alpha_i)
        acc.append( exp_i.eval('RF'))
        all_exp.append(exp_i)
    if(verbose):
        print('Acc:')
        print(acc)
    best=np.argmax(acc)
    return  alpha[best],all_exp[best]

def single_exp(data_path,hyper_path,prot_obj):
    print(data_path)
    dataset=data.get_data(data_path)
    hyper_dict=bayes_optim(dataset=dataset,
                           alg_params=base.AlgParams(),
                           protocol=prot_obj,
                           verbose=1)
    with open(hyper_path, 'w') as fp:
        json.dump(hyper_dict, fp)
    return hyper_dict

def multiple_exp(args,prot):
    if(args.multi):
        exp=utils.DirFun([("data_path",0),("hyper_path",1)])(single_exp)
    else:
        exp=single_exp
    exp(data_path=args.data,
        hyper_path=args.hyper,
        prot_obj=prot)

if __name__ == '__main__':
    parser =  utils.get_args(['data','hyper'],
                             ['n_split','n_iter'])
    args = parser.parse_args()
    prot=protocol.Protocol(io_type=protocol.NNetIO,
                           split_gen=protocol.SplitGenerator(n_split=args.n_split,
                                                             n_iters=args.n_iter))
    multiple_exp(args,
                 prot)