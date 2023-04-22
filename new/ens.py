import numpy as np
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
from keras import callbacks
from sklearn.base import BaseEstimator, ClassifierMixin

class NeuralEnsembleGPU(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.binary_builder=BinaryModel()
        self.multi_builder=None
        self.binary_model=None
        self.multi_model=None

    def fit(self,X,targets):
        data_params=get_dataset_params(X,targets)
        binary_full=self.binary_builder(data_params)
#        binary_full.summary()
        y_binary=binarize(targets)
        train_model(X,y_binary,binary_full,data_params)
        self.binary_model=Extractor(binary_full,data_params['n_cats'])
        binary=self.binary_model.predict(X)
        print(len(binary))
        raise NotImplementedError

    def predict_proba(self,X):
        raise NotImplementedError

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
        'batch_size':X.shape[0]}

def train_model(X,y,model,params):
    earlystopping = callbacks.EarlyStopping(monitor="accuracy",
                mode="min", patience=5,restore_best_weights=True)
    model.fit(X,y,epochs=100,batch_size=params['batch_size'],
        verbose = 0,callbacks=earlystopping)

class BinaryModel(object):
    def __init__(self,first=1.0,second=1):
        self.first=first
        self.second=second

    def __call__(self,params):
        first_hidden=int(self.first*params['dims'])
        second_hidden=int(self.second*params['dims'])
        input_layer = Input(shape=(params['dims']))
        models=[]
        for i in range(params['n_cats']):
            x_i=Dense(first_hidden,activation='relu',
            	name=f"first{i}")(input_layer)
            x_i=Dense(second_hidden,activation='relu',
            	name=f"hidden{i}")(x_i)
            x_i=BatchNormalization(name=f'batch{i}')(x_i)
            x_i=Dense(2, activation='softmax')(x_i)
            models.append(x_i)
        concat_layer = Concatenate()(models)
        model= Model(inputs=input_layer, outputs=concat_layer)
        model.compile(loss='categorical_crossentropy',
            optimizer='adam',metrics=['accuracy'])
        return model

def binarize(labels):
    n_cats=max(labels)+1
    y=[]
    for l_i in labels:
        vector_i=[]
        for j in range(n_cats):
            if(j==l_i):
                vector_i+=[1,0]
            else:
                vector_i+=[0,1]
        y.append(vector_i)
    return np.array(y)