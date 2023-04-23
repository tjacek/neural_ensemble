import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
from keras import callbacks
from tensorflow import one_hot
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import preprocessing
import learn

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

    def fit(self,X,targets,verbose=False):
        data_params=get_dataset_params(X,targets)
        binary_full=self.binary_builder(data_params)
        y_binary=binarize(targets)
        history=train_model(X,y_binary,binary_full,data_params)
        if(verbose):
            show_history(history)        
        self.binary_model=Extractor(binary_full,data_params['n_cats'])
        data_params['binary_dims']=self.binary_model.binary_dim()
        binary=self.binary_model.predict(X)
        self.multi_model=self.multi_builder(data_params)
        y=one_hot(targets,data_params['n_cats'])
        
        X_multi=[X]+binary
        y_multi=[y for i in range(data_params['n_cats'])]
        history=train_model(X_multi,y_multi,self.multi_model,data_params)
        if(verbose):
            show_history(history)        
        return self

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

    def __str__(self):
        params=f'binary:{self.binary_builder},multi:{self.multi_builder}'
        return f'NeuralEnsembleGPU({params})'

class NeuralEnsembleCPU(BaseEstimator, ClassifierMixin):
    def __init__(self,binary=None,multi_clf='RF'):
        if(binary is None):
            binary=BinaryBuilder()
        self.binary_builder=binary #BinaryBuilder()
        self.multi_clf=multi_clf
        self.clfs=[]

    def fit(self,X,targets,verbose=False):
        data_params=get_dataset_params(X,targets)
        binary_full=self.binary_builder(data_params)
        y_binary=binarize(targets)
        history=train_model(X,y_binary,binary_full,data_params)
        if(verbose):
            show_history(history)        
        self.binary_model=Extractor(binary_full,data_params['n_cats'])
        binary=self.binary_model.predict(X)
        for binary_i in binary:
            multi_i=np.concatenate([X,binary_i],axis=1)
            clf_i =learn.get_clf(self.multi_clf)
            clf_i.fit(multi_i,targets)
            self.clfs.append(clf_i)
        return self

    def predict_proba(self,X):
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

    def __str__(self):
        params=f'binary:{self.binary_builder},multi:{self.multi_clf}'
        return f'NeuralEnsembleCPU({params})'

def show_history(history):
    msg=''
    for key_i,value_i in history.history.items():
        if('accuracy' in key_i):
            msg+=f'{key_i}:{value_i[-1]:.2f} '
    print(msg)

def is_neural_ensemble(clf):
    return (isinstance(clf,NeuralEnsembleGPU) or 
               isinstance(clf,NeuralEnsembleCPU))

class Extractor(object):
    def __init__(self,full_model,n_cats=3):
        self.extractors=[]
        for i in range(n_cats):
            extractor_i=Model(inputs=full_model.input,
                outputs=full_model.get_layer(f'hidden{i}').output)
            self.extractors.append(extractor_i)
    
    def binary_dim(self):
        return self.extractors[-1].output.shape[1]

    def predict(self,X):
        binary=[]
        for extractor_i in self.extractors:
            binary_i=extractor_i.predict(X,verbose=0) 
            binary_i=preprocessing.scale(binary_i)
            binary.append(binary_i)
        return binary

def get_dataset_params(X,y):
    return {'n_cats':max(y)+1,'dims':X.shape[1],
        'batch_size': int(1.0*X.shape[0])}

def train_model(X,y,model,params):
    earlystopping = callbacks.EarlyStopping(monitor="accuracy",
                mode="min", patience=5,restore_best_weights=True)
    return model.fit(X,y,epochs=50,batch_size=params['batch_size'],
        verbose = 0,callbacks=None)#earlystopping)

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
        loss={f'binary{i}' :'categorical_crossentropy' 
                for i in range(params['n_cats'])}
        metrics={f'binary{i}' :'accuracy' 
                for i in range(params['n_cats'])}
        model= Model(inputs=input_layer, outputs=outputs)#concat_layer)
        model.compile(loss=loss,  #'mean_squared_error',
            optimizer='adam',metrics=metrics)
        return model

    def layer_name(self,i,j):
        if(j==(len(self.hidden)-1)):
            return f'hidden{i}'
        return f'{i}_{j}'

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

class MultiInputBuilder(object):
    def __init__(self,hidden=(1,1)):#first=1.5,second=0.33):
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
        loss={f'multi{i}' :'categorical_crossentropy' 
                for i in range(params['n_cats'])}
        metrics={f'multi{i}' :'accuracy' 
                for i in range(params['n_cats'])}

        model= Model(inputs=inputs, outputs=outputs)
        optim=tf.keras.optimizers.Adam(learning_rate=0.1)
#        optim=tf.keras.optimizers.RMSprop(learning_rate=0.00001)
        model.compile(loss=loss,
            optimizer=optim,metrics=metrics)
        return model

    def __str__(self):
        return '_'.join([str(h) for h in self.hidden])