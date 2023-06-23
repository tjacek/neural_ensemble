import tools
tools.silence_warnings()
import argparse
import numpy as np
import tensorflow as tf 
from sklearn.metrics import accuracy_score#,f1_score

import data,deep

def single_exp(data_path,hyper_path):
    X,y=data.get_dataset(data_path)
    hyper_params=parse_hyper(hyper_path)
    dataset_params=data.get_dataset_params(X,y)
    splits=data.gen_splits(X,y,n_splits=10,n_repeats=1)
    acc=[]
    for (X_train,y_train),(X_test,y_test) in splits():
        model=deep.simple_nn(dataset_params,hyper_params)
        y_train = tf.keras.utils.to_categorical(y_train, 
        	                num_classes = dataset_params['n_cats'])
        model.fit(X_train,y_train)
        y_pred= model.predict(X_test)
        y_pred=np.argmax(y_pred,axis=1)
        acc.append(accuracy_score(y_test,y_pred))
    print(acc)

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
    parser.add_argument("--data", type=str, default='data/wine-quality-red')
    parser.add_argument("--hyper", type=str, default='hyper.txt')
    args = parser.parse_args()
    single_exp(args.data,args.hyper)