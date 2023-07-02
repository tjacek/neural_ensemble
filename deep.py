import numpy as np
import tensorflow as tf
import keras
from sklearn import preprocessing
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

class EnsembleBuilder(object):
    def __init__(self,loss_type=0.5):
        self.loss=get_loss(loss_type)

    def __call__(self,params,hyper_params):
        input_layer = Input(shape=(params['dims']))
        single_cls=[]
        for i in range(params['n_cats']):
            nn_i=nn_builder(params,hyper_params,input_layer,False,i)
            single_cls.append(nn_i)
        binary_loss=BinaryLoss()
        loss={}
        class_dict=params['class_weights']
        for i in range(params['n_cats']):        
            loss[f'out_{i}']=binary_loss(i,class_dict)
        metrics={f'out_{i}' : 'accuracy'
                for i in range(params['n_cats'])}
        model= Model(inputs=input_layer, outputs=single_cls)
        model.compile(loss=loss, #loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=metrics)
        return BinaryEnsemble(model)#,params['n_cats'])

def get_loss(loss_type):
    if(type(loss_type)==float):
        return BinaryLoss(loss_type)
    def helper(i,class_dict):
        return 'categorical_crossentropy'
    return helper

class BinaryEnsemble(object):
    def __init__(self,multi_output):
        self.multi_output=multi_output
        self.n_clf=self.multi_output.output_shape[0][1]
        self.extractor=None 

    def fit(self,X,y,epochs=150,batch_size=None, verbose=0,callbacks=None):
        y_multi=[y for i in range(self.n_clf)]
        self.multi_output.fit(X,y_multi,batch_size=batch_size,epochs=epochs,
                            verbose=verbose,callbacks=callbacks)

    def predict(self,X):
        y_pred= self.multi_output.predict(X)
        n_samples,n_cats=y_pred[0].shape
        final_pred=[]
        for i in range(n_samples):
            ballot_i=np.array([y_pred[j][i] 
                for j in range(n_cats)])
            final_pred.append(np.sum(ballot_i,axis=0))
        return final_pred

    def extract(self,X):
        if(self.extractor is None):
            names=[ layer.name for layer in self.multi_output.layers]
            penult=names[-6:-3]
            layers=[self.multi_output.get_layer(name_i).output 
                    for name_i in penult]
            self.extractor=Model(inputs=self.multi_output.input,outputs=layers)
        return self.extractor.predict(X)

    def get_full(self,train,scale=True):
        cs_train=self.extract(train.X)
        if(scale):
            cs_train=[preprocessing.scale(cs_i)
                    for cs_i in cs_train]
        return [np.concatenate([train.X,cs_i],axis=1)
                for cs_i in cs_train]

    def save(self,out_path):
        self.multi_output.save(out_path)

class BinaryLoss(object):
    def __init__(self,alpha=0.5):
        self.alpha=alpha

    def __call__(self,i,class_dict):
        other_i=[ size_j  
                for cat_j,size_j in class_dict.items()
                    if(cat_j!=i)]
        cat_size_i  = self.alpha*(1/class_dict[i])
        other_size_i= (1.0-self.alpha) *  (1/sum(other_i))
        class_weights= [other_size_i for i in range(len(class_dict) )]
        class_weights[i]=cat_size_i
        class_weights=np.array(class_weights,dtype=np.float32)
        return weighted_binary_loss(class_weights)

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

def weighted_binary_loss( class_weights):
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

#def binary_weights(class_sizes,i):#,double=False ):
#    rest=[value_j for j,value_j in class_sizes.items()
#                  if(j!=i)]
#    rest= sum(rest)
#    weights=[1/rest, 1/class_sizes[i]]
#    return np.array(weights,dtype=np.float32)/sum(weights)