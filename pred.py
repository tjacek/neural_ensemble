import tools
tools.silence_warnings()
import argparse
import numpy as np
import tensorflow as tf 
from keras import callbacks
import json
import data,deep

def single_exp(data_path,model_path,out_path):
    X,y=data.get_dataset(data_path)
    for name_i,model_i,split_i in get_model_paths(model_path):
        test_X, test_y=split_i.get_test(X,y)
        y_pred=model_i.predict(test_X)
        y_pred=np.argmax(y_pred,axis=1)
        print(list(zip(test_y,y_pred)))



def get_model_paths(model_path):
    for path_i in tools.get_dirs(model_path):
        name_i=path_i.split('/')[-1]
        for model_j in tools.get_dirs(path_i):
            nn_j = tf.keras.models.load_model(f'{model_j}/nn',compile=False)
            if('ens' in name_i):
                nn_j=deep.BinaryEnsemble(nn_j,5)
            test_ind=np.load(f'{model_j}/test.npy')
            train_ind=np.load(f'{model_j}/train.npy')
            split_j=data.DataSplit(train_ind,test_ind)
            yield name_i,nn_j,split_j

def make_pred(make_model,out_path,params,splits):
    hyper_params,dataset_params=params
    earlystopping = callbacks.EarlyStopping(monitor='accuracy',#params['metric'],
                mode="max", patience=5,restore_best_weights=True)
    all_pred=[]
    for (X_train,y_train),(X_test,y_test) in splits():
        model=make_model(dataset_params,hyper_params)
        y_train = tf.keras.utils.to_categorical(y_train, 
        	                num_classes = dataset_params['n_cats'])
        model.fit(X_train,y_train,epochs=150,callbacks=earlystopping)
        y_pred= model.predict(X_test)
        y_pred=np.argmax(y_pred,axis=1)
        all_pred.append((y_test,y_pred))
    with open(out_path, 'wb') as f:
        all_pred=[ (test_i.tolist(),pred_i.tolist()) 
            for test_i,pred_i in all_pred]
        json_str = json.dumps(all_pred, default=str)         
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='data/cmc')
    parser.add_argument("--models", type=str, default='cmc')
    parser.add_argument("--pred", type=str, default='pred_cmc')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    if(args.dir>0):
        @tools.dir_fun#(single_exp)
        def helper(in_path,out_path):
            name_i=in_path.split('/')[-1]
            hyper_i=f'{args.hyper}/{name_i}'
            single_exp(in_path,hyper_i,out_path)
        helper(args.data,args.pred)
    else:
        single_exp(args.data,args.models,args.pred)