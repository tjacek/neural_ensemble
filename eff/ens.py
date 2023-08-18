import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
import deep,loss

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
    if(ens_type=='weighted'):
        return WeightedBuilder(0.5)
    raise Exception(f'Type {ens_type} unknown')


class MultiEns(deep.NeuralEnsemble):
    def __init__(self, model,params,hyper_params,split,ens_type='multi'):
        super().__init__(model,params,hyper_params,split)
        self.ens_type=ens_type

    def get_type(self):
        return self.ens_type#'multi'

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

    def get_penultimate(self,i,k):
        if(self.hyper_params['batch']):
            return f'batch_{i}_{k}'
        j=len(self.hyper_params['layers'])-1
        return f"layer_{i}_{k}_{j}"

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


class WeightedBuilder(object):
    def __init__(self,alpha=0.5):
        self.alpha=alpha

    def __call__(self,params,hyper_params,split):
        model=ens_builder(params,
                     hyper_params,
                     n_splits=len(split))
        metrics={f'output_{i}_{k}':'accuracy' 
                for i in range(len(split))
                    for k in range(params['n_cats'])}
        class_dict=params['class_weights']
        loss_dict={}
        for i in range(params['n_cats']):
            for k in range(len(split)):
                key_ik=f'output_{k}_{i}'
                loss_dict[key_ik]=loss.weighted_loss(i=i,
                                                     class_dict=class_dict,
                                                     alpha=self.alpha)
        model.compile(loss=loss_dict,
                      optimizer='adam',
                      metrics=metrics)
        deep_ens=MultiEns(model=model,
                          params=params,
                          hyper_params=hyper_params,
                          split=split,
                          ens_type='weighted')
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