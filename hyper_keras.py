import conf
conf.silence_warnings()
import numpy as np
import argparse
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
import keras_tuner as kt
import data,binary

class EnsmbleBuilder(object):
    def __init__(self,dim,n_cats,n_hidden,l1):
        self.dim=dim
        self.n_cats=n_cats
        self.n_hidden=n_hidden
        self.l1=l1

    def __call__(self,hp):
        input_layer = Input(shape=(self.dim))
        l1_coff = hp.Float('l1', min_value=self.l1[0], 
        	max_value=self.l1[0], step=32)
        #hp.Choice('kernel_regularizer', values=[0.01,0.001,0.1,0.005,0.05]) 
        p_units = hp.Int('units', min_value=self.n_hidden[0], 
        	max_value=self.n_hidden[1], step=32)
              
        models=[]
        for i in range(self.n_cats):
            x_i=Dense(units=p_units,activation='relu',name=f"hidden{i}",
            	kernel_regularizer=regularizers.l1(l1=l1_coff))(input_layer)
            x_i=BatchNormalization(name=f'batch{i}')(x_i)
            x_i=Dense(2,activation='softmax')(x_i)
            models.append(x_i)
        concat_layer = Concatenate()(models)
        model= Model(inputs=input_layer, outputs=concat_layer)
        optim=optimizers.RMSprop(learning_rate=0.00001)
        model.compile(loss='categorical_crossentropy',
            optimizer=optim)#,metrics=['accuracy'])
#        model.summary()
        return model

def hyper_exp(conf_dict,n_split):
    data.make_dir(conf_dict['main_dict'])
    print('Optimisation for hyperparams')
    for hyper_i in conf_dict['hyperparams']:
        hyper_values= ','.join(map(str,conf_dict[hyper_i]))
        print('{}:{}'.format(hyper_i,hyper_values))
    
    helper=lambda x:(min(x),max(x))
    hp_ranges={'l1':helper(conf_dict['l1']),
        'hid_ratio':helper(conf_dict['hid_ratio'])
    }

    for path_i in data.top_files(conf_dict['json']):
        print(f'Optimisation of hyperparams for dataset {path_i}')
        raw_data=data.read_data(path_i)
        best=single_exp(raw_data,hp_ranges)
        print(best)
        line_i=','.join([str(v) for v in best.values()])+'\n'
        with open(conf_dict['hyper'],"a") as f:
            f.write(line_i) 
    print('Hyperparams saved at {}'.format(conf_dict['hyper']))

def single_exp(raw_data,hp_ranges):
    dim=raw_data.dim()
    n_cats= raw_data.n_cats()
    l1,hid_ratio=hp_ranges['l1'],hp_ranges['hid_ratio']
    n_hidden=dim* np.array(hid_ratio)
    model_builder=EnsmbleBuilder(dim,n_cats,n_hidden,l1)
    tuner = kt.Hyperband(model_builder,
                objective="categorical_crossentropy",#'val_accuracy',
                max_epochs=100,
                factor=3,
                directory=None,#'my_dir',
                project_name=None)#'intro_to_kt')
    X,y,names=raw_data.as_dataset()
    y=np.array(binary.binarize(y))
    tuner.search(X, y, epochs=50, validation_split=0.1)#, callbacks=[stop_early])
    best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
    best={'l1':best_hps.get('l1'),'units':best_hps.get('units')}
    return best

if __name__ == "__main__":
    args=conf.parse_args(default_conf='conf/small.cfg') 
    conf_dict=conf.read_conf(args.conf,
        ['dir','hyper','clf'],args.dir_path)
#    print(conf_dict)
    hyper_exp(conf_dict,args.n_split)