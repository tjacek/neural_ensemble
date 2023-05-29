import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
from keras import callbacks
from tensorflow import one_hot
#from sklearn.utils import class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import preprocessing
import learn,tools

class NeuralEnsembleGPU(BaseEstimator, ClassifierMixin):
    def __init__(self,binary=None,multi=None):
        if(binary is None):
            binary=BinaryBuilder()
        if(multi is None):
            multi=MultiInputBuilder()
        self.binary_builder=binary
        self.multi_builder=multi
        self.binary_model=None
        self.multi_model=None
        self.data_params=None

    def fit(self,X,targets,verbose=False):
        data_params=get_dataset_params(X,targets)
        self.data_params=data_params
        binary_full=self.binary_builder(data_params)
        y_binary=binarize(targets)
        history=train_model(X,y_binary,binary_full,data_params)
        binary_acc= accuracy_desc(history)
        if(verbose):
            show_history(history)        
        self.binary_model=Extractor(binary_full,data_params['n_cats'])
        data_params['binary_dims']=self.binary_model.binary_dim()
        binary=self.binary_model.predict(X)
        self.multi_model=self.multi_builder(data_params)
        y=targets 
        
        X_multi=[X]+binary
        y_multi=[y for i in range(data_params['n_cats'])]

        history=train_model(X_multi,y_multi,self.multi_model,data_params,
           None)
        multi_acc= accuracy_desc(history)
        if(verbose):
            show_history(history)        
        return {**binary_acc,**multi_acc }

    def predict_proba(self,X):
        binary=self.binary_model.predict(X)
        X_multi=[X]+binary
        y=self.multi_model.predict(X_multi,verbose=0)
        y=np.array(y)
        prob=np.sum(y,axis=0)
        return prob

    def predict(self,X):
        prob=self.predict_proba(X)
        return np.argmax(prob,axis=1)

    def empty_model(self):
        binary_full=self.binary_builder(self.data_params)
        self.binary_model=Extractor(binary_full,self.data_params['n_cats'])
        self.multi_model=self.multi_builder(self.data_params)

    def load_weights(self,in_path):
        self.binary_model.load_weights(f'{in_path}/binary')
        self.multi_model.load_weights(f'{in_path}/multi.h5')

    def save_weights(self,out_path):
        tools.make_dir(out_path)
        self.binary_model.save_weights(f'{out_path}/binary')
        self.multi_model.save_weights(f'{out_path}/multi.h5')

    def __str__(self):
        params=f'binary:{self.binary_builder},multi:{self.multi_builder}'
        return f'NeuralEnsembleGPU({params})'

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

    def fit(self,X,targets,verbose=False):
        data_params=get_dataset_params(X,targets)
        self.data_params=data_params
        binary_full=self.binary_builder(data_params)
        y_binary=binarize(targets)
        history=train_model(X,y_binary,binary_full,data_params)
        if(verbose):
            show_history(history,self.data_params)        
        self.binary_model=Extractor(binary_full,data_params['n_cats'])
        self.train_data=(X,targets)
        self.catch= accuracy_desc(history,self.data_params)
        return self.catch

    def train_clfs(self,train_data,multi_clf=None):
        if(multi_clf is None):
            multi_clf=self.multi_clf
        self.clfs=[]
        X,targets=train_data
        binary=self.binary_model.predict(X)
        for binary_i in binary:
            multi_i=np.concatenate([X,binary_i],axis=1)
            clf_i =learn.get_clf(multi_clf)
            clf_i.fit(multi_i,targets)
            self.clfs.append(clf_i)

    def predict_proba(self,X):
        if(len(self.clfs)==0):
            self.train_clfs(self.train_data)
            self.train_data=None
        binary=self.binary_model.predict(X)
        votes=[]
        for i,binary_i in enumerate(binary):
            multi_i=np.concatenate([X,binary_i],axis=1)
            vote_i= self.clfs[i].predict_proba(multi_i)
            votes.append(vote_i)
        votes=np.array(votes)
        prob=np.sum(votes,axis=0)
        return prob
    
    def predict(self,X):
        prob=self.predict_proba(X)
        return np.argmax(prob,axis=1)
    
    def empty_model(self):
        binary_full=self.binary_builder(self.data_params)
        self.binary_model=Extractor(binary_full,self.data_params['n_cats'])

    def load_weights(self,in_path):
        self.binary_model.load_weights(f'{in_path}/binary')

    def save_weights(self,out_path):
        tools.make_dir(out_path)
        self.binary_model.save_weights(f'{out_path}/binary')

    def __str__(self):
        params=f'binary:{self.binary_builder},multi:{self.multi_clf}'
        return f'NeuralEnsembleCPU({params})'

class Extractor(object):
    def __init__(self,full_model,n_cats=3):
        self.extractors=[]
        for i in range(n_cats):
            extractor_i=Model(inputs=full_model.input,
                outputs=full_model.get_layer(f'hidden{i}').output)
            self.extractors.append(extractor_i)
    
    def binary_dim(self):
        return self.extractors[-1].output.shape[1]

    def predict(self,X,scale=True):
        binary=[]
        for extractor_i in self.extractors:
            binary_i=extractor_i.predict(X,verbose=0) 
            if(scale):
                binary_i=preprocessing.scale(binary_i)
            binary.append(binary_i)
        return binary
    
    def load_weights(self,in_path):
        for i,path_i in enumerate(tools.top_files(in_path)):
            self.extractors[i].load_weights(path_i)

    def save_weights(self,out_path):
        tools.make_dir(out_path)
        for i,extr_i in enumerate(self.extractors):
            extr_i.save_weights(f'{out_path}/{i}.h5') 

class BinaryBuilder(object):
    def __init__(self,hidden=(1,0.5)):
        self.hidden=hidden

    def __call__(self,params):
        input_layer = Input(shape=(params['dims']))
        outputs=[]
        x_i=input_layer
        for i in range(params['n_cats']):
            for j,hidden_j in enumerate(self.hidden):
                hidden_j=int(hidden_j*params['dims'])
                name_j=self.layer_name(i,j)
                x_i=Dense(hidden_j,activation='relu',
                    name=name_j)(x_i)
            x_i=Dense(2, activation='softmax',name=f'binary{i}')(x_i)
            outputs.append(x_i)

#        class_weight= list(params['class_weights'].values() )
        loss={f'binary{i}' : weighted_binary_loss(params['class_weights'],i)#'binary_crossentropy' 
                for i in range(params['n_cats'])}
        metrics={f'binary{i}' :  get_metric(params['metric'])  #'accuracy'
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

class MultiInputBuilder(object):
    def __init__(self,hidden=(1,1)):
        self.hidden=hidden

    def __call__(self,params):
        common= Input(shape=(params['dims']))
        inputs,outputs=[common],[]
        for i in range(params['n_cats']):
            input_i = Input(shape=(params['binary_dims']))
            inputs.append(input_i)
            x_i=Concatenate()([common,input_i])
            for j,hidden_j in enumerate(self.hidden):
                hidden_j=int(hidden_j*params['dims'])
                x_i=Dense(hidden_j,activation='relu',
                    name=f"{i}_{j}")(x_i)
#            x_i=BatchNormalization(name=f'batch{i}')(x_i)
            x_i=Dense(params['n_cats'], activation='softmax',
                name=f'multi{i}')(x_i)
            outputs.append(x_i)
        
        custom_loss= weighted_categorical_crossentropy(
            list(params['class_weights'].values() ))
        loss={f'multi{i}' : custom_loss #'categorical_crossentropy' 
                for i in range(params['n_cats'])}
        metrics={f'multi{i}' : params['metric']#'accuracy' 
                for i in range(params['n_cats'])}

        model= Model(inputs=inputs, outputs=outputs)
        optim=tf.keras.optimizers.Adam(learning_rate=0.1)
        model.compile(loss=loss,
            optimizer=optim,metrics=metrics)
        return model

    def __str__(self):
        return '_'.join([str(h) for h in self.hidden])

def binarize(labels):
    n_cats=max(labels)+1
    binary_labels=[]
    for i in range(n_cats):
        y_i=[int(y_i==i) for y_i in labels]
        y_i=one_hot(y_i,2)
        binary_labels.append(y_i)
    return binary_labels

def show_history(history,params):
    metric_name='balanced_accuracy' #str(params['metric'].metric_name ).lower()
    msg=''
    for key_i,value_i in history.history.items():
        if(metric_name in key_i):
            msg+=f'{key_i}:{value_i[-1]:.2f} '
    print(msg)

def accuracy_desc(history,params):
    metric_name='balanced_accuracy' #str(params['metric'].metric_name).lower()
    acc_desc={}
    for key_i in history.history:
        if(metric_name in key_i):
            acc_i=history.history[key_i][-1]
            acc_desc[key_i]=round(acc_i,4)
    return acc_desc

def is_neural_ensemble(clf):
    return (isinstance(clf,NeuralEnsembleGPU) or 
               isinstance(clf,NeuralEnsembleCPU))

def get_dataset_params(X,y):
    param_dict={'n_cats':max(y)+1,'dims':X.shape[1],
        'batch_size': int(1.0*X.shape[0])}
    class_weights={cat_i:1 for cat_i in range(param_dict['n_cats'])}
    for i in y:
        class_weights[i]+=1
    param_dict['class_weights']={ key_i:value_i#(1.0/value_i) 
        for key_i,value_i in class_weights.items()}
    param_dict['metric']= 'balanced_accuracy' #'Precision'
    return param_dict

def get_metric(name_i):
    if(name_i=='balanced_accuracy'):
        return BalancedAccuracy()
    return name_i

def train_model(X,y,model,params,weights=None):
    earlystopping = callbacks.EarlyStopping(monitor=params['metric'],
                mode="max", patience=5,restore_best_weights=True)
    return model.fit(X,y,epochs=50,batch_size=params['batch_size'],
        verbose = 0,callbacks=earlystopping,class_weight=None)

def weighted_categorical_crossentropy(class_weight):
    def loss(y_obs,y_pred):
        y_obs = tf.dtypes.cast(y_obs,tf.int32)
        hothot = tf.one_hot(tf.reshape(y_obs,[-1]), depth=len(class_weight))
        weight = tf.math.multiply(class_weight,hothot)
        weight = tf.reduce_sum(weight,axis=-1)
        losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels=y_obs, logits=y_pred,weights=weight
        )
        return losses
    return loss

def weighted_binary_loss( class_sizes,i):
    class_weights= binary_weights(class_sizes,i)
    def loss(y_obs,y_pred):        
        y_obs = tf.dtypes.cast(y_obs,tf.int32)
#        hothot = tf.one_hot(tf.reshape(y_obs,[-1]), depth=len(class_weights))
        hothot=  tf.dtypes.cast( y_obs,tf.float32)

        weights = tf.math.multiply(class_weights,hothot)

        weights = tf.reduce_sum(weights,axis=-1)
        y_obs= tf.argmax(y_obs,axis=1)

        losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels=y_obs, logits=y_pred,weights=weights
        )
        return losses
    return loss

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

def binary_weights(class_sizes,i,double=False ):
    rest=[value_j for j,value_j in class_sizes.items()
                  if(j!=i)]
#    if(double):
#        rest=[1/r for r in rest]
    rest= sum(rest)
    weights=[1/rest, 1/class_sizes[i]]
#    weights=[]
#    for k in class_sizes:
#        if(k==i):
#            weights.append( 1/class_sizes[k])
#        else:
#            weights.append(1/rest)
    return np.array(weights,dtype=np.float32)/sum(weights)