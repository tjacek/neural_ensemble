import tools
tools.silence_warnings()
import argparse
import numpy as np
import tensorflow as tf 
import json
import data,deep

def single_exp(data_path,hyper_path,out_path):
    X,y=data.get_dataset(data_path)
    hyper_params=parse_hyper(hyper_path)
    dataset_params=data.get_dataset_params(X,y)
    splits=data.gen_splits(X,y,n_splits=10,n_repeats=1)
    alg_dict={'base':deep.simple_nn,'binary':deep.binary_ensemble}
    tools.make_dir(out_path)
    for name_i,make_model_i in alg_dict.items():
        out_i=f'{out_path}/{name_i}'
        make_pred(make_model_i,out_i,hyper_params,dataset_params,splits)

def make_pred(make_model,out_path,hyper_params,dataset_params,splits):
    all_pred=[]
    for (X_train,y_train),(X_test,y_test) in splits():
        model=make_model(dataset_params,hyper_params)
        y_train = tf.keras.utils.to_categorical(y_train, 
        	                num_classes = dataset_params['n_cats'])
        model.fit(X_train,y_train)
        y_pred= model.predict(X_test)
        y_pred=np.argmax(y_pred,axis=1)
        all_pred.append((y_test,y_pred))
    with open(out_path, 'wb') as f:
        all_pred=[ (test_i.tolist(),pred_i.tolist()) 
            for test_i,pred_i in all_pred]
        json_str = json.dumps(all_pred, default=str)         
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)

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
    parser.add_argument("--data", type=str, default='data')#/wine-quality-red')
    parser.add_argument("--hyper", type=str, default='hyper')
    parser.add_argument("--pred", type=str, default='pred')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    if(args.dir>0):
        @tools.dir_fun#(single_exp)
        def helper(in_path,out_path):
            name_i=in_path.split('/')[-1]
            hyper_i=f'{args.hyper}/{name_i}'
            print(in_path)
            print(hyper_i)
            print(out_path)
            single_exp(in_path,hyper_i,out_path)
        helper(args.data,args.pred)
    else:
        single_exp(args.data,args.hyper,args.pred)