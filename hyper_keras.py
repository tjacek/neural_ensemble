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


class SimpleBuilder(object):#kt.HyperModel):
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
        
#        raise Exception(self.n_hidden)
        p_units = hp.Int('hid_ratio', min_value=int(self.n_hidden[0]), 
            max_value= int(self.n_hidden[1]), step=32)
              
        x_i=Dense(units=p_units,activation='relu',name=f"hidden",
                kernel_regularizer=regularizers.l1(l1=l1_coff))(input_layer)
        x_i=BatchNormalization(name=f'batch')(x_i)
        x_i=Dense(self.n_cats,activation='softmax')(x_i)

        model= Model(inputs=input_layer, outputs=x_i)
        optim=optimizers.RMSprop(learning_rate=0.00001)
        model.compile(loss='categorical_crossentropy',
            optimizer=optim,metrics=['accuracy'])
#        model.summary()
        return model

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
        p_units = hp.Int('n_hidden', min_value= self.n_hidden[0], 
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
            optimizer=optim,metrics=['accuracy'])
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
    names=conf_dict['hyperparams']
    with open(conf_dict['hyper'],"a") as f:
        f.write('dataset,{}\n'.format(','.join(names))) 
    for path_i in data.top_files(conf_dict['json']):
        print(f'Optimisation of hyperparams for dataset {path_i}')
        raw_data=data.read_data(path_i)
        dim=raw_data.dim()
        split_ratio=1.0/args.n_split
        best=single_exp(raw_data,hp_ranges,split_ratio)
        best['hid_ratio']= best['hid_ratio']/float(dim)
        print(best)
        data_i=path_i.split('/')[-1]
        line_i='{},{}\n'.format(data_i,
           ','.join([str(best[name_j])  for name_j in names]))
        line_i=','.join([str(v) for v in best.values()])+'\n'
        with open(conf_dict['hyper'],"a") as f:
            f.write(line_i) 
    print('Hyperparams saved at {}'.format(conf_dict['hyper']))

def single_exp(raw_data,hp_ranges,split_ratio=0.1):
    dim=raw_data.dim()
    n_cats= raw_data.n_cats()
    l1,hid_ratio=hp_ranges['l1'],hp_ranges['hid_ratio']
    n_hidden=dim* np.array(hid_ratio)#*10
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    model_builder= SimpleBuilder(dim,n_cats,n_hidden,l1) #EnsmbleBuilder(dim,n_cats,n_hidden,l1)

    tuner=kt.BayesianOptimization(model_builder,
                objective='accuracy',
                max_trials=4,
                overwrite=True)
    X,y,names=raw_data.as_dataset()
    y=tf.one_hot(y,depth=n_cats)
#    y=np.array(binary.binarize(y))

    tuner.search(X, y, epochs=150, validation_split=split_ratio,
       verbose=0,callbacks=[stop_early])
    tuner.results_summary()
    best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
    best={'l1':best_hps.get('l1'),'hid_ratio':best_hps.get('hid_ratio')}
    return best

if __name__ == "__main__":
    args=conf.parse_args(default_conf='conf/small.cfg') 
    conf_dict=conf.read_conf(args.conf,
        ['dir','hyper','clf'],args.dir_path)
#    print(conf_dict)
    hyper_exp(conf_dict,args.n_split)