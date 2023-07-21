import numpy as np
import tensorflow as tf
import keras
from sklearn import preprocessing
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model

class NeuralEnsemble(object):
    def __init__(self,multi_output,prepare_labels,ens_type,n_clf):
        self.multi_output=multi_output
        self.prepare_labels=prepare_labels
        self.ens_type=ens_type
        self.n_clf=n_clf #self.multi_output.output_shape[0][1]
        self.extractor=None 

    def fit(self,X,y,epochs=150,batch_size=None, verbose=0,callbacks=None):
        y_multi=self.prepare_labels(y,self.n_clf)
        return self.multi_output.fit(X,y_multi,
                            batch_size=batch_size,epochs=epochs,
                            verbose=verbose,callbacks=callbacks)

    def predict(self,X):
        y_pred= self.multi_output.predict(X,verbose=0)
        n_samples,n_cats=y_pred[0].shape
        final_pred=[]
        for i in range(n_samples):
            ballot_i=np.array([y_pred[j][i] 
                for j in range(n_cats)])
            final_pred.append(np.sum(ballot_i,axis=0))
        return final_pred

    def extract(self,X,scale=False):
        if(self.extractor is None):
            names=[ layer.name for layer in self.multi_output.layers]
            penult=names[-6:-3]
            layers=[self.multi_output.get_layer(name_i).output 
                    for name_i in penult]
            self.extractor=Model(inputs=self.multi_output.input,outputs=layers)
        feats= self.extractor.predict(X,verbose=0)
        if(scale):
            feats=[preprocessing.scale(feat_i)
                      for feat_i in feats]
        return feats

    def get_full(self,train,scale=True):
        if isinstance(train,np.ndarray):
            X=train
        else:
            X=train.X
        cs_train=self.extract(X,scale)
        return [np.concatenate([X,cs_i],axis=1)
                for cs_i in cs_train]
    
    def load_weights(self,in_path):
        self.multi_output.load_weights(in_path)

    def save_weights(self,out_path):
        self.multi_output.save_weights(out_path)

    def save(self,out_path):
        self.multi_output.save(out_path)

    def __str__(self):
        return f'type:{self.ens_type}\nn_clf:{self.n_clf}'

class EnsembleFactory(object):
    def __init__(self,loss_fun,labels,ens_type,output_cats=None):
        self.loss_fun=loss_fun
        self.labels=labels
        self.ens_type=ens_type
        self.output_cats=output_cats

    def __call__(self,params,hyper_params):
        if(self.output_cats is None):
            self.output_cats=params['n_cats']
        model=ensemble_builder(params,hyper_params,
                self.loss_fun,output_cats=self.output_cats)
        return NeuralEnsemble(model,
                              self.labels,
                              self.ens_type,
                              params['n_cats'])      
def get_ensemble(ens_type):
    if(ens_type=='base'):
        return simple_nn
    if(ens_type=='multi'):
        return multi_ensemble()
    if(type(ens_type)==tuple):
        ens_type,alpha=ens_type
        if(ens_type=='weighted'):
            return weighted_ensemble(alpha)
        elif(ens_type=='binary'):
            return binary_ensemble(alpha)
    raise Exception(f"Ensemble {ens_type} not implemented") 

def multi_ensemble():
    def loss_fun(i,class_dict):
        return 'categorical_crossentropy'
    return EnsembleFactory(loss_fun=loss_fun,
                           labels=basic_labels,
                           ens_type='multi',
                           output_cats=None)

def weighted_ensemble(alpha=0.5):
    def loss_fun(i,class_dict):
        return weighted_loss(i,class_dict,alpha)
    return EnsembleFactory(loss_fun=loss_fun,
                           labels=basic_labels,
                           ens_type='weighted',
                           output_cats=None)

def binary_ensemble(alpha=0.5):
    def loss_fun(i,class_dict):
        return binary_loss(i,class_dict,alpha)
    return EnsembleFactory(loss_fun=loss_fun,
                           labels=binary_labels,
                           ens_type='binary',
                           output_cats=2)

def ensemble_builder(params,hyper_params,binary_loss,output_cats=2):
    input_layer = Input(shape=(params['dims']))
    class_dict=params['class_weights']
    single_cls,loss,metrics=[],{},{}
    for i in range(params['n_cats']):
        nn_i=nn_builder(params,hyper_params,input_layer,
            as_model=False,i=i,n_cats=output_cats)
        single_cls.append(nn_i)
        loss[f'out_{i}']=binary_loss(i,class_dict)
        metrics[f'out_{i}']= 'accuracy'
    model= Model(inputs=input_layer, outputs=single_cls)
    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=metrics)
    return model

#def read_ensemble(in_path):
#    print(in_path)
#    nn = tf.keras.models.load_model(f'{in_path}/nn.h5',compile=False)
#    with open(f'{in_path}/ens_desc',"r") as f:
#        raw= f.read().split('\n')
#        ens_type=raw[0].split(':')[-1]
#        n_clf=int(raw[1].split(':')[-1])
#        if(ens_type=='binary'):
#            return NeuralEnsemble(nn,binary_labels,'binary',n_clf)
#        if(ens_type=='weighted'):
#            return NeuralEnsemble(nn,basic_labels,'weighted',n_clf)
#        return NeuralEnsemble(nn,basic_labels,'multi',n_clf)

def basic_labels(y,n_clf):
    return [y for i in range(n_clf)]

def binary_labels(y,n_clf):
    binary_y=[]
    for cat_i in range(n_clf):
        y_i=[y_j[cat_i] for y_j in y]
        y_i = tf.keras.utils.to_categorical(y_i, 
                            num_classes = 2)
        binary_y.append(y_i)
    return binary_y

def simple_nn(params,hyper_params):
    model=nn_builder(params,hyper_params)
    model.compile(loss='categorical_crossentropy', 
            optimizer='adam',metrics='accuracy')
    return model

def nn_builder(params,hyper_params,input_layer=None,as_model=True,i=0,n_cats=None):
    if(input_layer is None):
        input_layer = Input(shape=(params['dims']))
    if(n_cats is None):
        n_cats=params['n_cats']
    x_i=input_layer#Concatenate()([common,input_i])
    for j,hidden_j in enumerate(hyper_params['layers']):
        x_i=Dense(hidden_j,activation='relu',
                    name=f"layer_{i}_{j}")(x_i)
    if(hyper_params['layers']):
        x_i=BatchNormalization(name=f'batch_{i}')(x_i)
    x_i=Dense(n_cats, activation='softmax',name=f'out_{i}')(x_i)
    if(as_model):
        return Model(inputs=input_layer, outputs=x_i)
    return x_i

def binary_loss(i,class_dict,alpha):
    one_i=class_dict[i]
    other_i=sum(class_dict.values())-one_i
    cat_size_i  = alpha*(1/one_i)
    other_size_i= (1.0-alpha) * (1/other_i)
    class_weights=[other_size_i,cat_size_i]
    class_weights=np.array(class_weights,dtype=np.float32)
    return weighted_binary_loss(class_weights)

def weighted_loss(i,class_dict,alpha):
    one_i=class_dict[i]
    other_i=sum(class_dict.values())-one_i
    cat_size_i  = alpha*(1/one_i)
    other_size_i= (1.0-alpha) * (1/other_i)
    class_weights= [other_size_i for i in range(len(class_dict) )]
    class_weights[i]=cat_size_i
    class_weights=np.array(class_weights,dtype=np.float32)
    return weighted_binary_loss(class_weights)

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