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

def nn_builder(params,hyper_params):
    input_layer = Input(shape=(params['dims']))         
    x_i=input_layer#Concatenate()([common,input_i])
    for j,hidden_j in enumerate(hyper_params['layers']):
        x_i=Dense(hidden_j,activation='relu',
                    name=f"layer_{j}")(x_i)
    if(hyper_params['layers']):
        x_i=BatchNormalization(name=f'batch')(x_i)
    x_i=Dense(params['n_cats'], activation='softmax',name=f'out')(x_i)
    model= Model(inputs=input_layer, outputs=x_i)
    return model
    
class NeuralEnsembleCPU(BaseEstimator, ClassifierMixin):
    def __init__(self,binary=None,multi_clf='RF'):
        if(binary is None):
            binary=BinaryBuilder()
        self.binary_builder=binary
        self.binary_model=None
        self.multi_clf=multi_clf
        self.clfs=[]
        self.train_data=None
        self.data_params=None
        self.catch=None


class BinaryBuilder(object):
    def __init__(self,hyper:dict):
        self.hyper=hyper

    def __call__(self,params):
        input_layer = Input(shape=(params['dims']))
        outputs=[]
        x_i=input_layer
        for i in range(params['n_cats']):
            outputs.append(build_model(self.hyper))
        loss={f'binary{i}' : weighted_binary_loss(params['class_weights'],i)
                for i in range(params['n_cats'])}
        metrics={f'binary{i}' :  get_metric(params['metric'])
                for i in range(params['n_cats'])}
        model= Model(inputs=input_layer, outputs=outputs)
        model.compile(loss=loss, 
            optimizer='adam',metrics=metrics)
        return model

    def layer_name(self,i,j):
        if(j==(len(self.hidden)-1)):
            return f'hidden{i}'
        return f'{i}_{j}'

    def __str__(self):
        return '_'.join([str(h) for h in self.hidden])

def build_model(hyper:dict):
    pass

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

#    for j,hidden_j in enumerate(self.hidden):
#        hidden_j=int(hidden_j*params['dims'])
#            name_j=self.layer_name(i,j)
#                x_i=Dense(hidden_j,activation='relu',
#                    name=name_j)(x_i)
#            x_i=BatchNormalization(name=f'batch{i}')(x_i)
#            x_i=Dense(2, activation='softmax',name=f'binary{i}')(x_i)
#            outputs.append(x_i)