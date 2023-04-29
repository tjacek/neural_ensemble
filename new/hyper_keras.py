import tools
tools.silence_warnings()
import numpy as np
import argparse,os
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
#from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
from time import time
import pandas as pd
import keras_tuner as kt
import ens

class BinaryKTBuilder(object):
    def __init__(self,params,hidden=[(0.25,5),(0.25,5)]):
        self.params=params
        self.hidden=hidden

    def __call__(self,hp):
        input_layer = Input(shape=(self.params['dims']))
        outputs=[]
        x_i=input_layer
        p=[]
        for j,(min_j,max_j) in enumerate(self.hidden):
            p_j=hp.Int(f'n_hidden{j}', 
                min_value= int(self.params['dims']*min_j), 
                max_value=int(self.params['dims']*max_j), 
                sampling="log",
                step=2)
            p.append(p_j)
        for i in range(self.params['n_cats']):
            for j,(min_j,max_j) in enumerate(self.hidden):
                name_j=self.layer_name(i,j)
                x_i=Dense(units=p[j],activation='relu',name=name_j)(x_i)
            x_i=Dense(2, activation='softmax',name=f'binary{i}')(x_i)
            outputs.append(x_i)
        loss={f'binary{i}' :'categorical_crossentropy' 
                for i in range(self.params['n_cats'])}
        metrics={f'binary{i}' :'accuracy' 
                for i in range(self.params['n_cats'])}
        model= Model(inputs=input_layer, outputs=outputs)
        model.compile(loss=loss,optimizer='adam',metrics=metrics)
        return model

    def get_params_names(self):
        return [f'n_hidden{j}' for j in range(len(self.hidden))]              

    def layer_name(self,i,j):
        if(j==(len(self.hidden)-1)):
            return f'hidden{i}'
        return f'{i}_{j}'
 
def single_exp(data_path,hyper_path,n_split,n_iter,n_bayes=20):#,ens_types):
    df=pd.read_csv(data_path) 
    X,y=tools.prepare_data(df)
    data_params=ens.get_dataset_params(X,y)
    print(data_params)
    model_builder= BinaryKTBuilder(data_params) 

    tuner=kt.BayesianOptimization(model_builder,
                objective='val_loss', #'accuracy']
                max_trials=n_bayes,
                overwrite=True)
    binary_y=ens.binarize(y)
    validation_split= 1.0/n_split
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X, binary_y, epochs=150, validation_split=validation_split,
       verbose=1,callbacks=[stop_early])
#    tuner.results_summary()
    best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
#    raise Exception(best_hps)
    best={ name_j: (best_hps.get(name_j)/ data_params['dims'])
           for name_j in model_builder.get_params_names()}
    print(best)
    return best

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='uci/wine-quality-red')
    parser.add_argument("--hyper", type=str, default='hyper.txt')
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=2)
#    parser.add_argument("--clfs", type=str, default='GPUClf_2_2,CPUClf_2')
    parser.add_argument("--log_path", type=str, default='log.time')

    args = parser.parse_args()
    tools.start_log(args.log_path)
    start=time()
    single_exp(args.data,args.hyper,args.n_split,
        args.n_iter)#,clf_types)
    tools.log_time(f'HYPER:{args.data}',start) 