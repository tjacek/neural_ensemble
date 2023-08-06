import tools
tools.silence_warnings()
import numpy as np
import tensorflow as tf
import keras
from sklearn import preprocessing
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
from keras import callbacks
import gzip
import data

class NeuralEnsemble(object):
    def __init__(self,model,params,hyper_params,split):
        self.model=model
        self.params=params
        self.hyper_params=hyper_params
        self.split=split
        self.pred_models=None

    def fit(self,x,y,batch_size,epochs=150,verbose=0,callbacks=None):
        raise NotImplementedError
    
    def predict(self,x,verbose=0):
        raise NotImplementedError

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


class BaseNN(NeuralEnsemble):
    def __init__(self, model,params,hyper_params,split):
        super().__init__(model,params,hyper_params,split)

    def fit(self,x,y,batch_size,epochs=150,verbose=0,callbacks=None):
        X,y=self.split.get_data(x,y,train=True)
        y=[ tf.keras.utils.to_categorical(y_i) 
              for y_i in y]
        self.model.fit(x=X,
                       y=y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=callbacks)

    def predict(self,x,verbose=0):
        X=self.split.get_data(x,train=False)
        if(self.pred_models is None):    
            self.pred_models=[]
            for i,x_i in enumerate(X):
                input_i= self.model.get_layer(f'input_{i}')
                output_i= self.model.get_layer(f'output_{i}')
                model_i=Model(inputs=input_i.input,
                              outputs=output_i.output)
                self.pred_models.append(model_i)
        y=[]
        for i,x_i in enumerate(X):
            y_i=self.pred_models[i].predict(x_i,
                                            verbose=verbose)
            y.append(y_i)
        y=np.concatenate(y,axis=0)
        return y

def read_ens(in_path):
    split=data.read_split(f'{in_path}/splits')
    weights=[]
    for path_i in tools.top_files(f'{in_path}/weights'):
        weights.append(np.load(path_i)['arr_0'] )
    with open(f'{in_path}/params',"r") as f:
        params= eval(f.read())
    with open(f'{in_path}/hyper_params',"r") as f:
        hyper_params= eval(f.read())
    deep_ens=build_ensemble(params,hyper_params,split)
    deep_ens.model.set_weights(weights)
#    raise Exception(dir(deep_ens.model))
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


def build_ensemble(params,hyper_params,split):
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

def train(in_path):
    dataset=data.get_dataset(in_path)
    all_splits=dataset.get_splits()

    params=dataset.get_params()
    hyper_params={'layers':[20,20],'batch':True}

    deep_ens=build_ensemble(params,hyper_params,all_split[0])
    early_stop = callbacks.EarlyStopping(monitor='accuracy',
                                         mode="max", 
                                         patience=5,
                                         restore_best_weights=True)
    deep_ens.fit(x=dataset.X,
                 y=dataset.y,
                 epochs=150,
                 batch_size=params['batch'],
                 verbose=1,
                 callbacks=early_stop)

    y_pred=deep_ens.predict_classes(x=dataset.X)
    acc=tools.get_metric('acc')
    print(acc(y_pred,dataset.y))

if __name__ == "__main__":
    in_path='../../uci/cleveland'
    train(in_path)