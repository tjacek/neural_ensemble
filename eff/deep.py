import numpy as np
import tensorflow as tf
import keras
from sklearn import preprocessing
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
import data

class NeuralEnsemble(object):
    def __init__(self,model):
        self.model=model
        self.split=None

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

#def train(in):

if __name__ == "__main__":
    in_path='../../uci/cleveland'
    full=data.get_dataset(in_path)
    params=full.get_params()
    hyper_params={'layers':[20,20],'batch':True}
    model=nn_builder(params,hyper_params,n_splits=10)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='accuracy')
    X=[full.X for i in range(10)]
    y=[tf.keras.utils.to_categorical(full.y, 
                                     num_classes = params['n_cats'])
            for i in range(10)]
    model.fit(X,y)