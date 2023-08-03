import numpy as np
import tensorflow as tf
import keras
from sklearn import preprocessing
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
from keras import callbacks
import data

class NeuralEnsemble(object):
    def __init__(self,model):
        self.model=model
        self.split=None

    def fit(self,X,y,batch_size,epochs=150,verbose=0,callbacks=None):
        X,y=self.split.get_train(X,y)
        y=[ tf.keras.utils.to_categorical(y_i) 
              for y_i in y]
        self.model.fit(x=X,
                       y=y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=callbacks)

def nn_builder(params,hyper_params,n_splits=10):
    outputs,inputs=[],[]
    for i in range(n_splits):
        input_layer = Input(shape=(params['dims']))
        inputs.append(input_layer)
        x_i=input_layer
        for j,hidden_j in enumerate(hyper_params['layers']):
            x_i=Dense(hidden_j,activation='relu',
                    name=f"layer_{i}_{j}")(x_i)
        if(hyper_params['batch']):
            x_i=BatchNormalization(name=f'batch_{i}')(x_i)
        x_i=Dense(params['n_cats'], activation='softmax',name=f'out_{i}')(x_i)
        outputs.append(x_i)
    return Model(inputs=inputs, outputs=outputs)

def train(in_path):
    dataset=data.get_dataset(in_path)
    all_splits=dataset.get_splits()

    params=dataset.get_params()
#    raise Exception(all_splits[0].get_sizes())
    hyper_params={'layers':[20,20],'batch':True}
    model=nn_builder(params,hyper_params,n_splits=10)
    metrics={f'out_{i}':'accuracy' for i in range(10)}
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=metrics)
    deep_ens=NeuralEnsemble(model)
    deep_ens.split=all_splits[0]

    early_stop = callbacks.EarlyStopping(monitor='accuracy',
                                         mode="max", 
                                         patience=5,
                                         restore_best_weights=True)
    deep_ens.fit(X=dataset.X,
                 y=dataset.y,
                 epochs=150,
                 batch_size=params['batch'],
                 verbose=1,
                 callbacks=early_stop)  
      
if __name__ == "__main__":
    in_path='../../uci/cleveland'
    train(in_path)