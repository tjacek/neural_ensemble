import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
from sklearn.base import BaseEstimator, ClassifierMixin

def simple_nn(params,hyper_params):
    model=nn_builder(params,hyper_params)
    model.compile(loss='categorical_crossentropy', 
            optimizer='adam',metrics='accuracy')
    return model

def nn_builder(params,hyper_params,input_layer=None,as_model=True,i=0):
    if(input_layer is None):
        input_layer = Input(shape=(params['dims']))         
    x_i=input_layer#Concatenate()([common,input_i])
    for j,hidden_j in enumerate(hyper_params['layers']):
        x_i=Dense(hidden_j,activation='relu',
                    name=f"layer_{i}_{j}")(x_i)
    if(hyper_params['layers']):
        x_i=BatchNormalization(name=f'batch_{i}')(x_i)
    x_i=Dense(params['n_cats'], activation='softmax',name=f'out_{i}')(x_i)
    if(as_model):
        return Model(inputs=input_layer, outputs=x_i)
    return x_i

def binary_ensemble(params,hyper_params):
    input_layer = Input(shape=(params['dims']))
    single_cls=[]
    for i in range(params['n_cats']):
        nn_i=nn_builder(params,hyper_params,input_layer,False,i)
        single_cls.append(nn_i)
        
#    loss={f'binary{i}' : weighted_binary_loss(params['class_weights'],i)#'binary_crossentropy' 
#                for i in range(params['n_cats'])}
    metrics={f'out_{i}' : 'accuracy'
                for i in range(params['n_cats'])}
    model= Model(inputs=input_layer, outputs=single_cls)
    model.compile(loss='categorical_crossentropy', #loss=loss, 
            optimizer='adam',metrics=metrics)
    return BinaryEnsemble(model,params['n_cats'])

class BinaryEnsemble(object):
    def __init__(self,multi_output,n_clf):
        self.multi_output=multi_output
        self.n_clf=n_clf

    def fit(self,X,y):
        y_multi=[y for i in range(self.n_clf)]
        self.multi_output.fit(X,y_multi)

    def predict(self,X):
        y_pred= self.multi_output.predict(X)
        n_samples,n_cats=y_pred[0].shape
        print((n_samples,n_cats))
        final_pred=[]
        for i in range(n_samples):
            ballot_i=np.array([y_pred[j][i] 
                for j in range(n_cats)])
            final_pred.append(np.sum(ballot_i,axis=0))
        return final_pred

def get_metric(name_i):
    if(name_i=='balanced_accuracy'):
        return BalancedAccuracy()
    return name_i

class BalancedAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_accuracy ', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = tf.argmax(y_true,axis=1)
        y_true_int = tf.cast(y_flat, tf.int32)
        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_flat, y_pred, sample_weight=weight)

def weighted_binary_loss( class_sizes,i):
    class_weights= binary_weights(class_sizes,i)
    def loss(y_obs,y_pred):        
        y_obs = tf.dtypes.cast(y_obs,tf.int32)
        hothot=  tf.dtypes.cast( y_obs,tf.float32)
        weights = tf.math.multiply(class_weights,hothot)
        weights = tf.reduce_sum(weights,axis=-1)
        y_obs= tf.argmax(y_obs,axis=1)
        losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels=y_obs, logits=y_pred,weights=weights
        )
        return losses
    return loss