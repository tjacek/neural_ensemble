import tools
tools.silence_warnings()
import numpy as np
import tensorflow as tf
import keras
from sklearn import preprocessing
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
from keras import callbacks
import data

#def get_penultimate(i,k,hyper_dict):
#    if(('batch' in hyper_dict) and hyper_dict['batch']):
#        return f'batch_{i}' #_{k}'
#    j=len(hyper_dict['layers'])
#    return f"layer_{i}_{k}_{j}"

class NeuralEnsemble(object):
    def __init__(self,model,params,hyper_params,split):
        self.model=model
        self.params=params
        self.hyper_params=hyper_params
        self.split=split
        self.pred_models=None
        self.extractors=None
    
    def __len__(self):
        return len(self.split)

    def get_type(self):
        raise NotImplementedError

    def fit(self,x,y,batch_size,epochs=150,verbose=0,callbacks=None):
        raise NotImplementedError
    
    def predict(self,x,verbose=0):
        raise NotImplementedError

    def get_penultimate(self,i,k):
        if(self.hyper_params['batch']):
            return f'batch_{i}_{k}'
        j=len(self.hyper_params['layers'])-1
        return f"layer_{i}_{k}_{j}"

    def extract(self,x,verbose=0):
        if(self.extractors is None):    
            self.extractors=[]
            for i in range(len(self)):
                out_names=[self.get_penultimate(i,k)#,self.hyper_params)
                            for k in range(self.params['n_cats'])]
                model_i=split_models(model=self.model,
                                          in_names=f'input_{i}',
                                          out_names=out_names)
                self.extractors.append(model_i)
        feats=[]
        for i in range(len(self)):
            feat_i=self.extractors[i].predict(x=x,
                                              verbose=verbose)
            
            feats.append(feat_i)
        return feats

    def predict_classes(self,x,verbose=0):
        prob= self.predict(x,verbose=verbose)
        return np.argmax(prob,axis=1)

    def save(self,out_path):
        tools.make_dir(out_path)
        self.split.save(f'{out_path}/splits')
        weights=self.model.get_weights()
        tools.make_dir(f'{out_path}/weights')
        for i,weight_i in enumerate(weights):
            np.savez_compressed(f'{out_path}/weights/{i}',
                                weight_i)          
        with open(f'{out_path}/params',"a") as f:
            f.write(f'{str(self.params)}') 
        with open(f'{out_path}/hyper_params',"a") as f:
            f.write(f'{str(self.hyper_params)}') 
        with open(f'{out_path}/type',"a") as f:
            f.write(self.get_type()) 

class BaseNN(NeuralEnsemble):
    def __init__(self, model,params,hyper_params,split):
        super().__init__(model,params,hyper_params,split)

    def get_type(self):
        return 'base'

    def fit(self,x,y,batch_size,epochs=150,verbose=0,callbacks=None):
        X,y=self.split.get_all(x,y,train=True)
        y=[ tf.keras.utils.to_categorical(y_i) 
              for y_i in y]
        self.model.fit(x=X,
                       y=y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=callbacks)

    def predict(self,x,verbose=0):
        X=self.split.get_all(x,train=False)
        if(self.pred_models is None):    
            self.pred_models=[]
            for i,x_i in enumerate(X):
                model_i=split_models(model=self.model,
                                     in_names=f'input_{i}',
                                     out_names=f'output_{i}')
                self.pred_models.append(model_i)
        y=[]
        for i,x_i in enumerate(X):
            y_i=self.pred_models[i].predict(x_i,
                                            verbose=verbose)
            y.append(y_i)
        y=np.concatenate(y,axis=0)
        return y

    def get_penultimate(self,i,k):
#        if(('batch' in hyper_dict) and 
        if(self.hyper_params['batch']):
            return f'batch_{i}' #_{k}'
        j=len(self.hyper_params['layers'])
        return f"layer_{i}_{k}_{j}"

def read_deep(in_path,builder=None):
    split=data.read_split(f'{in_path}/splits')
    weights=[]
    for path_i in tools.top_files(f'{in_path}/weights'):
        weights.append(np.load(path_i)['arr_0'] )
    with open(f'{in_path}/params',"r") as f:
        params= eval(f.read())
    with open(f'{in_path}/hyper_params',"r") as f:
        hyper_params= eval(f.read())
    if(builder is None):
        builder=make_base
    deep_ens=builder(params,hyper_params,split)
    deep_ens.model.set_weights(weights)
    return deep_ens

def make_base(params,hyper_params,split):
    model=nn_builder(params,hyper_params,n_splits=len(split))
    metrics={f'output_{i}':'accuracy' 
        for i in range(len(split))}
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=metrics)
    deep_ens=BaseNN(model=model,
                    params=params,
                    hyper_params=hyper_params,
                    split=split)
    return deep_ens

def nn_builder(params,hyper_params,n_splits=10):
    outputs,inputs=[],[]
    for i in range(n_splits):
        input_layer = Input(shape=(params['dims']),
                            name=f'input_{i}')
        inputs.append(input_layer)
        x_i=input_layer
        for j,hidden_j in enumerate(hyper_params['layers']):
            x_i=Dense(hidden_j,
                      activation='relu',
                      name=f"layer_{i}_{j}")(x_i)
        if(hyper_params['batch']):
            x_i=BatchNormalization(name=f'batch_{i}')(x_i)
        x_i=Dense(params['n_cats'], 
                  activation='softmax',
                  name=f'output_{i}')(x_i)
        outputs.append(x_i)
    return Model(inputs=inputs, outputs=outputs)

def split_models(model,in_names,out_names):
    if(type(in_names)==str):
        inputs= model.get_layer(in_names).input
    else:
        inputs=[ model.get_layer(name_i).input
                   for name_i in in_names]
    if(type(out_names)==str):
        outputs= self.model.get_layer(out_names).output
    else:
        outputs=[ model.get_layer(name_i).output
                   for name_i in out_names]
    return Model(inputs=inputs,
                 outputs=outputs)