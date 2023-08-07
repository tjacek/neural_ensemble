import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
import deep

def read_ens(in_path):    
    with open(f'{in_path}/type',"r") as f:
        ens_type= f.read()
    builder=get_builder(ens_type)    
    return deep.read_deep(in_path=in_path,
                          builder=builder)

def get_builder(ens_type:str):
    if(ens_type=='base'):
        return deep.make_base
    if(ens_type=='multi'):
        return build_multi
    raise Exception(f'Type {ens_type} unknown')

def get_penultimate(i,k,hyper_dict):
    if(hyper_dict['batch']):
        return f'batch_{i}_{k}'
    j=len(hyper_dict['layers'])
    return f"layer_{i}_{k}_{j}"

class MultiEns(deep.NeuralEnsemble):
    def __init__(self, model,params,hyper_params,split):
        super().__init__(model,params,hyper_params,split)

    def get_type(self):
        return 'multi'

    def fit(self,x,y,batch_size,epochs=150,verbose=0,callbacks=None):
        X,y=self.split.get_all(x,y,train=True)
        y=[ tf.keras.utils.to_categorical(y_i) 
              for k in range(self.params['n_cats'])
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
            for i in range(len(self)):
                out_names=[ f'output_{i}_{k}'
                            for k in range(self.params['n_cats'])     ]
                model_i=deep.split_models(model=self.model,
                                          in_names=f'input_{i}',
                                          out_names=out_names)
                self.pred_models.append(model_i)
        y=[]
        for i,x_i in enumerate(X):
            y_i=self.pred_models[i].predict(x_i,
                                            verbose=verbose)
            y_i=np.array(y_i)
            y_i=np.sum(y_i,axis=0)
            y.append(y_i)
        y=np.concatenate(y,axis=0)
        return y

    def extract(self,x,verbose=0):
        if(self.extractors is None):    
            self.extractors=[]
            for i in range(len(self)):
                out_names=[get_penultimate(i,k,self.hyper_params)
                            for k in range(self.params['n_cats'])]
                model_i=deep.split_models(model=self.model,
                                          in_names=f'input_{i}',
                                          out_names=out_names)
                self.extractors.append(model_i)
        feats=[]
        for i in range(len(self)):
            feat_i=self.extractors[i].predict(x=x,
                                              verbose=verbose)
            
            feats.append(feat_i)
        return feats

def build_multi(params,hyper_params,split):
    model=ens_builder(params,
                     hyper_params,
                     n_splits=len(split))
    metrics={f'output_{i}_{k}':'accuracy' 
                for i in range(len(split))
                    for k in range(params['n_cats'])}
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=metrics)
    deep_ens=MultiEns(model=model,
                       params=params,
                       hyper_params=hyper_params,
                       split=split)
    return deep_ens

def ens_builder(params,hyper_params,n_splits=10):
    outputs,inputs=[],[]
    for i in range(n_splits):
        input_layer = Input(shape=(params['dims']),
                            name=f'input_{i}')
        inputs.append(input_layer)
        for k in range(params['n_cats']): 
            x_i=input_layer
            for j,hidden_j in enumerate(hyper_params['layers']):
                x_i=Dense(hidden_j,
                          activation='relu',
                          name=f"layer_{i}_{k}_{j}")(x_i)
            if(hyper_params['batch']):
                x_i=BatchNormalization(name=f'batch_{i}_{k}')(x_i)
            x_i=Dense(params['n_cats'], 
                      activation='softmax',
                      name=f'output_{i}_{k}')(x_i)
            outputs.append(x_i)
    return Model(inputs=inputs, outputs=outputs)