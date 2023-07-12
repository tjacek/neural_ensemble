import tools
tools.silence_warnings()
import argparse
import numpy as np
import tensorflow as tf 
from keras import callbacks
import data,deep

class ExpFactory(object):
    def __init__(self,dataset_params,hyper_params):
        self.dataset_params=dataset_params
        self.hyper_params=hyper_params	
        self.early_stop = callbacks.EarlyStopping(monitor='accuracy',
                                                  mode="max", 
                                                  patience=5,
                                                  restore_best_weights=True)

    def __call__(self,X,y,split,ens_type):
        make_model=deep.get_ensemble(ens_type)
        model=make_model(self.dataset_params,self.hyper_params)
        X_train,y_train=split.get_train(X,y)
        y_train = tf.keras.utils.to_categorical(y_train, 
        	                                    num_classes = self.dataset_params['n_cats'])
        history=model.fit(X_train,y_train,
        	              batch_size=self.dataset_params['batch'],
                          epochs=150,
                          verbose=0,
                          callbacks=self.early_stop)
        return Exp(split,
        	       model,
        	       ens_type,
        	       self.hyper_params,
        	       self.dataset_params)

class Exp(object):
    def __init__(self,split,model,ens_type,hyper,data):
        self.split=split
        self.model=model
        self.ens_type=ens_type
        self.hyper=hyper
        self.data=data

    def save(self,out_path):
        self.model.save_weights(f'{out_path}/weights')
        np.save(f'{out_path}/train',self.split.train_ind)
        np.save(f'{out_path}/test',self.split.test_ind)
        with open(f'{out_path}/info',"a") as f:
            f.write(f'{self.ens_type}\n')
            f.write(f'{str(self.hyper)}\n') 
            f.write(f'{str(self.data)}\n') 

@tools.log_time(task='TRAIN')
def single_exp(data_path,hyper_path,out_path,n_splits=10,n_repeats=10):
    print(data_path)
    X,y=data.get_dataset(data_path)
    hyper_params=parse_hyper(hyper_path)
    dataset_params=data.get_dataset_params(X,y)
    exp_factory=ExpFactory(dataset_params,hyper_params)
    all_splits=data.gen_splits(X,y,
    	                       n_splits=n_splits,
    	                       n_repeats=n_repeats)
#    alg_dict={ 'base':deep.simple_nn,
#               'multi-ens':deep.multi_ensemble(),
#               'binary-ens-0.25':deep.binary_ensemble(0.25)
#             }
    algs=['base','multi',('binary',0.25)]
    tools.make_dir(out_path)
    for alg_i in algs:
        name_i=get_name(alg_i)
        tools.make_dir(f'{out_path}/{name_i}')
        for j,split_j in enumerate(all_splits.splits):
            out_j=f'{out_path}/{name_i}/{j}'
            exp_j= exp_factory(X,y,split_j,alg_i)
            exp_j.save(out_j)

def get_name(alg_i):
    if(type(alg_i)==tuple):
        return f'{alg_i[0]}-{str(alg_i[1])}'
    return alg_i

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
    parser.add_argument("--data", type=str, default='../data')
    parser.add_argument("--hyper", type=str, default='../test3/hyper')
    parser.add_argument("--models", type=str, default='../test3/models')
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--dir", type=int, default=0)
    parser.add_argument("--log", type=str, default='../test3/log.info')
    args = parser.parse_args()
    tools.start_log(args.log)
    if(args.dir>0):
        single_exp=tools.dir_fun(3)(single_exp)
    single_exp(args.data,args.hyper,args.models,args.n_splits,args.n_repeats)