import tools
tools.silence_warnings()
import argparse
import numpy as np
import tensorflow as tf 
from keras import callbacks
import data,deep

@tools.log_time(task='TRAIN')
def single_exp(data_path,hyper_path,out_path,n_splits=10,n_repeats=10):
    X,y=data.get_dataset(data_path)
    hyper_params=parse_hyper(hyper_path)
    dataset_params=data.get_dataset_params(X,y)
    splits=data.gen_splits(X,y,n_splits=n_splits,n_repeats=n_repeats)
    alg_dict={'base':deep.simple_nn,
              'multi_ens':deep.EnsembleBuilder('multi'),
              'binary_ens(0.5)':deep.EnsembleBuilder(0.5),
             }
    earlystopping = callbacks.EarlyStopping(monitor='accuracy',
                mode="max", patience=5,restore_best_weights=True)
    def train_model(X_train,y_train,make_model):
        model=make_model(dataset_params,hyper_params)
        y_train = tf.keras.utils.to_categorical(y_train, 
        	                num_classes = dataset_params['n_cats'])
        batch=dataset_params['batch']
        model.fit(X_train,y_train,batch_size=batch,epochs=150,
            verbose=0,callbacks=earlystopping)
        return model
    tools.make_dir(out_path)
    for name_i,make_model_i in alg_dict.items():
        tools.make_dir(f'{out_path}/{name_i}')
        for j,((train_ind,train_data),test) in enumerate(splits()):
            out_j=f'{out_path}/{name_i}/{j}'
            tools.make_dir(out_j)
            model_j=train_model(*train_data,make_model_i)
            model_j.save(f'{out_j}/nn')
            np.save(f'{out_j}/train',train_ind)
            np.save(f'{out_j}/test',test[0])

def parse_hyper(hyper_path):
    with open(hyper_path) as f:
        line = eval(f.readlines()[-1])
        hyper_dict=line[0]
        layers= [key_i for key_i in hyper_dict
                   if('unit' in key_i)]
        layers.sort()
        return { 'batch':hyper_dict['batch'],
                 'layers':[hyper_dict[name_j] 
                            for name_j in layers] }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='data')
    parser.add_argument("--hyper", type=str, default='hyper')
    parser.add_argument("--models", type=str, default='models')
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--dir", type=int, default=0)
    parser.add_argument("--log", type=str, default='log.info')
    args = parser.parse_args()
    tools.start_log(args.log)
    if(args.dir>0):
        single_exp=tools.dir_fun(3)(single_exp)
    single_exp(args.data,args.hyper,args.models,args.n_splits,args.n_repeats)