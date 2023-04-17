import os
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
from keras import callbacks
from tensorflow import one_hot
from sklearn.base import BaseEstimator, ClassifierMixin
import conf,learn

class BinaryEnsemble(object):
    def __init__(self,n_hidden=10,l1=0.001):
        self.n_hidden=n_hidden
        self.l1=l1
        self.optim=optimizers.RMSprop(learning_rate=0.00001)

    def __call__(self,params):
        input_layer = Input(shape=(params['dims']))
        if(self.l1>0):
            reg=regularizers.l1(0.001)
        else:
            reg=None
        models=[]
        for i in range(params['n_cats']):
            x_i=Dense(self.n_hidden,activation='relu',name=f"hidden{i}",
                kernel_regularizer=reg)(input_layer)
            x_i=BatchNormalization(name=f'batch{i}')(x_i)
            x_i=Dense(2, activation='softmax')(x_i)
            models.append(x_i)
        concat_layer = Concatenate()(models)
        model= Model(inputs=input_layer, outputs=concat_layer)
        model.compile(loss='categorical_crossentropy',
            optimizer=self.optim,metrics=['accuracy'])
#        model.summary()
        return model

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
        batch_size= conf.GLOBAL['batch_size']#int(conf.GLOBAL['batch_ratio']* X.shape[0])
        earlystopping = callbacks.EarlyStopping(monitor="accuracy",
                mode="min", patience=5,restore_best_weights=True)
        nn_params={'dims':X.shape[1],'n_cats':n_cats}
        self.model=SimpleNN(n_hidden=n_hidden)(nn_params)
        y=one_hot(targets,n_cats)
        self.model.fit(X,y,epochs=500,batch_size=batch_size,
            verbose = 0,callbacks=earlystopping)

    def predict_proba(self,X):
        return self.model.predict(X)

    def predict(self,X):
        prob=self.model.predict(X)
        return np.argmax(prob,axis=1)

def unified_binary(train_names,datasets):
    X,y,n_cats=[],[],len(datasets)
    for name_i in train_names:
        x_i=[data_j[name_i] for data_j in datasets]
        cat_i=name_i.get_cat()
        y_i=[one_hot(cat_i,n_cats) for j in range(n_cats)]
        x_i=np.concatenate(x_i,axis=0)
        y_i=np.concatenate(y_i,axis=0)
        X.append(x_i)
        y.append(y_i)
    X,y=np.array(X),np.array(y)
    return X,y

class MulticlassNN(BaseEstimator, ClassifierMixin):
    def __init__(self,ratio=0.10):
        self.ratio=ratio
        self.model=None

    def __call__(self,datasets):
        sample_dataset=datasets[0]
        train,test=sample_dataset.split()
        train_names,test_names=train.keys(),test.keys()
        X_train,y_train=unified_binary(train_names,datasets)
        n_cats=len(datasets)
        self.fit(X_train,y_train,n_cats)
        X_test,y_test=unified_binary(test_names,datasets)
        raw_pred=self.model.predict(X_test)
        
        raw_pred= np.array(np.split(raw_pred, n_cats, axis=1))
        votes=np.sum(raw_pred,axis=0)
        y_pred=np.argmax(votes,axis=1)
        return learn.make_result(test_names,y_pred)

    def fit(self,X,targets,n_cats=10):
#        n_cats=max(targets)+1
        n_hidden= int(self.ratio*X.shape[1])   
        batch_size=X.shape[0] #conf.GLOBAL['batch_size']
        params={'n_cats':n_cats,'dims':int(X.shape[1]/n_cats),
            'n_hidden':n_hidden}
        self.model=self.build_model(params)
#        self.model.summary()
        earlystopping = callbacks.EarlyStopping(monitor="accuracy",
                mode="min", patience=5,restore_best_weights=True)
        self.model.fit(X,targets,epochs=500,batch_size=batch_size,
            verbose = 0,callbacks=earlystopping)

    def build_model(self,params):
        model = Sequential()
        dims=params['dims']

        input_dim=params['n_cats']*params['dims']
        input_layer=tensorflow.keras.layers.Input(shape=input_dim)
        
        layer_splits=tf.split(input_layer, 
            num_or_size_splits=params['n_cats'], axis=1)
        prob_layers=[]
        for i,input_i in enumerate(layer_splits):
            x_i=Dense(params['n_hidden'],activation='relu',name=f"hidden{i}",
                kernel_regularizer=None)(input_i)
            x_i=BatchNormalization()(x_i)
            x_i=Dense(params['n_cats'], activation='softmax')(x_i)        
            prob_layers.append(x_i)
        concat_layer = Concatenate()(prob_layers)
        model= Model(inputs=input_layer, outputs=concat_layer)
        optim=optimizers.RMSprop(learning_rate=0.00001)
        model.compile(loss='mean_squared_error', #'categorical_crossentropy',
            optimizer=optim,metrics=['accuracy'])
        return model