import os
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model

class SimpleNN(object):
    def __init__(self,n_hidden=10,l1=0.001):
        self.n_hidden=n_hidden
        self.l1=l1
        self.optim=optimizers.RMSprop(learning_rate=0.00001)

    def __call__(self,params):
        model = Sequential()
        if(self.l1>0):
            reg=regularizers.l1(0.001)
        else:
            reg=None
        model.add(Dense(self.n_hidden, input_dim=params['dims'], activation='relu',name="hidden",
            kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Dense(params['n_cats'], activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=self.optim, 
            metrics=['accuracy'])
#        model.summary()
        return model

def get_extractor(model_i):
    return Model(inputs=model_i.input,
                outputs=model_i.get_layer('hidden').output)