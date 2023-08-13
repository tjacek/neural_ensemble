import tensorflow as tf
import numpy as np

#def get_loss(ens_type):
#    if(ens_type=='multi'):
#        def loss_fun(i,class_dict):
#            return 'categorical_crossentropy'
#    if(type(ens_type)==tuple):
#        ens_type,alpha=ens_type
#        if(ens_type=='weighted'):
#            def loss_fun(i,class_dict):
#                return weighted_loss(i,class_dict,alpha)
#        elif(ens_type=='binary'):
#            def loss_fun(i,class_dict):
#                return binary_loss(i,class_dict,alpha)
#    return loss_fun

def binary_loss(i,class_dict,alpha):
    one_i=class_dict[i]
    other_i=sum(class_dict.values())-one_i
    cat_size_i  = alpha*(1/one_i)
    other_size_i= (1.0-alpha) * (1/other_i)
    class_weights=[other_size_i,cat_size_i]
    class_weights=np.array(class_weights,dtype=np.float32)
    return keras_loss(class_weights)

def weighted_loss(i,class_dict,alpha):
    one_i=class_dict[i]
    other_i=sum(class_dict.values())-one_i
    cat_size_i  = alpha*(1/one_i)
    other_size_i= (1.0-alpha) * (1/other_i)
    class_weights= [other_size_i for i in range(len(class_dict) )]
    class_weights[i]=cat_size_i
    class_weights=np.array(class_weights,dtype=np.float32)
    return keras_loss(class_weights)

def keras_loss( class_weights):
    def loss(y_obs,y_pred):        
        y_obs = tf.dtypes.cast(y_obs,tf.int32)
        hothot=  tf.dtypes.cast( y_obs,tf.float32)
        weights = tf.math.multiply(class_weights,hothot)
        weights = tf.reduce_sum(weights,axis=-1)
        y_obs= tf.argmax(y_obs,axis=1)
        losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels=y_obs, 
            logits=y_pred,
            weights=weights
        )
        return losses
    return loss