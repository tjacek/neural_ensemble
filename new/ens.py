import numpy as np
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
from keras import callbacks
from tensorflow import one_hot
from sklearn.base import BaseEstimator, ClassifierMixin

class NeuralEnsembleGPU(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.binary_builder=BinaryBuilder()
        self.multi_builder=MultiInputBuilder()
        self.binary_model=None
        self.multi_model=None

    def fit(self,X,targets):
        data_params=get_dataset_params(X,targets)
#        raise Exception(data_params)
        binary_full=self.binary_builder(data_params)
#        binary_full.summary()
        y_binary=binarize(targets)
        history=train_model(X,y_binary,binary_full,data_params)
        print(history.history['loss'])
        self.binary_model=Extractor(binary_full,data_params['n_cats'])
        binary=self.binary_model.predict(X)
        print('binary')
        self.multi_model=self.multi_builder(data_params)
        y=one_hot(targets,data_params['n_cats'])

#        full=[X]+binary
        history=train_model(binary,y,self.multi_model,data_params)
        print(history.history['loss'])
        print('multiclass')
        return self

    def predict_proba(self,X):
        binary=self.binary_model.predict(X)
        y=self.multi_model.predict(binary,verbose=0)
        return y

    def predict(self,X):
        prob=self.predict_proba(X)
        return np.argmax(prob,axis=1)

class Extractor(object):
    def __init__(self,full_model,n_cats=3):
        self.extractors=[]
        for i in range(n_cats):
            extractor_i=Model(inputs=full_model.input,
                outputs=full_model.get_layer(f'hidden{i}').output)
            self.extractors.append(extractor_i)

    def predict(self,X):
        return [extractor_i.predict(X,verbose=0) 
            for extractor_i in self.extractors]

def get_dataset_params(X,y):
    return {'n_cats':max(y)+1,'dims':X.shape[1],
        'batch_size': int(1.0*X.shape[0])}

def train_model(X,y,model,params):
    earlystopping = callbacks.EarlyStopping(monitor="accuracy",
                mode="min", patience=5,restore_best_weights=True)
    return model.fit(X,y,epochs=50,batch_size=params['batch_size'],
        verbose = 2,callbacks=earlystopping)

class BinaryBuilder(object):
    def __init__(self,first=1.0,second=1.0):
        self.first=first
        self.second=second

    def __call__(self,params):
        first_hidden=int(self.first*params['dims'])
        second_hidden=int(self.second*params['dims'])
        input_layer = Input(shape=(params['dims']))
        outputs=[]
        for i in range(params['n_cats']):
            x_i=Dense(first_hidden,activation='relu',
            	name=f"first{i}")(input_layer)
            x_i=Dense(second_hidden,activation='relu',
            	name=f"hidden{i}")(x_i)
#            x_i=BatchNormalization(name=f'batch{i}')(x_i)
            x_i=Dense(2, activation='softmax',name=f'binary{i}')(x_i)
            outputs.append(x_i)
#        concat_layer = Concatenate()(models)
        loss={f'binary{i}' :'categorical_crossentropy' 
                for i in range(params['n_cats'])}
        metrics={f'binary{i}' :'accuracy' 
                for i in range(params['n_cats'])}
        model= Model(inputs=input_layer, outputs=outputs)#concat_layer)
        model.compile(loss=loss,  #'mean_squared_error',
            optimizer='adam',metrics=metrics)
        return model

def binarize(labels):
    n_cats=max(labels)+1
    binary_labels=[]
    for i in range(n_cats):
        y_i=[int(y_i==i) for y_i in labels]
        y_i=one_hot(y_i,2)
        binary_labels.append(y_i)
    return binary_labels

class MultiInputBuilder(object):
    def __init__(self,first=1.0,second=1.0):
        self.first=first
        self.second=second

    def __call__(self,params):
        first_hidden=int(self.first*params['dims'])
        second_hidden=int(self.second*params['dims'])
#        common= Input(shape=(params['dims']))
        inputs,outputs=[],[]
        for i in range(params['n_cats']):
            input_i = Input(shape=(params['dims']))
            inputs.append(input_i)
#            concat_i=
            x_i=Dense(first_hidden,activation='relu',
                name=f"first{i}")(input_i)
            x_i=Dense(second_hidden,activation='relu',
                name=f"hidden{i}")(x_i)
#            x_i=BatchNormalization(name=f'batch{i}')(x_i)
#            x_i=Dense(params['n_cats'], activation='softmax')(x_i)
            outputs.append(x_i)
        concat_layer = Concatenate()(outputs)
        tmp=Dense(params['n_cats'], activation='softmax')(concat_layer)
        model= Model(inputs=inputs, outputs=tmp)
        model.compile(loss='categorical_crossentropy',
            optimizer='adam',metrics=['accuracy'])
        return model