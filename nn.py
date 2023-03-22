import os
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
from keras import callbacks
from tensorflow import one_hot
from sklearn.base import BaseEstimator, ClassifierMixin

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

class NNFacade(BaseEstimator, ClassifierMixin):
    def __init__(self,ratio=0.33):
        self.ratio=ratio
        self.model=None

    def fit(self,X,targets):
        n_cats=max(targets)+1
        n_hidden= int(self.ratio*X.shape[1])
        earlystopping = callbacks.EarlyStopping(monitor="accuracy",
                mode="min", patience=5,restore_best_weights=True)
        nn_params={'dims':X.shape[1],'n_cats':n_cats}
        self.model=SimpleNN(n_hidden=n_hidden)(nn_params)
        y=one_hot(targets,n_cats)
        self.model.fit(X,y,epochs=500,
            batch_size=32,verbose = 0,callbacks=earlystopping)

    def predict_proba(self,X):
        return self.model.predict(X)

    def predict(self,X):
        prob=self.model.predict(X)
        return np.argmax(prob,axis=1)