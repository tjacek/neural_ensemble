import conf
conf.silence_warnings()
import argparse
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
import keras_tuner as kt
import data

class EnsmbleBuilder(object):
    def __init__(self,dims,n_cats,n_hidden):
        self.dims=dims
        self.n_cats=n_cats
        self.n_hidden=n_hidden

    def __call__(self,hp):
        input_layer = Input(shape=(self.dimd))
        l1_coff = hp.Float('l1', min_value=0, max_value=0.001, step=32)
        #hp.Choice('kernel_regularizer', values=[0.01,0.001,0.1,0.005,0.05]) 
        p_units = hp.Int('units', min_value=10, max_value=150, step=32)
              
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

def hyper_exp(conf_path,n_split):
    data.make_dir(conf_dict['main_dict'])
    print('Optimisation for hyperparams')
    for hyper_i in conf_dict['hyperparams']:
        hyper_values= ','.join(map(str,conf_dict[hyper_i]))
        print('{}:{}'.format(hyper_i,hyper_values))
    for path_i in data.top_files(conf_dict['json']):
        print(f'Optimisation of hyperparams for dataset {path_i}')
        raw_data=data.read_data(path_i)
        single_exp(raw_data)

def single_exp(raw_data):
    dim=raw_data.dim()
    n_cats= raw_data.n_cats()
    print((dim,n_cats))

if __name__ == "__main__":
    args=conf.parse_args(default_conf='conf/small.cfg') 
    conf_dict=conf.read_conf(args.conf,
        ['dir','hyper','clf'],args.dir_path)
#    print(conf_dict)
    hyper_exp(conf_dict,args.n_split)