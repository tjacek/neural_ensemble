from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
from sklearn.base import BaseEstimator, ClassifierMixin

class NeuralEnsembleGPU(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.binary_builder=BinaryModel()
        self.multi_builder=None
        self.binary_model=None
        self.multi_model=None

    def fit(self,X,targets):
        data_params=get_dataset_params(X,targets)
        self.binary_model=self.binary_builder(data_params)
        raise NotImplementedError

    def predict_proba(self,X):
        raise NotImplementedError

    def predict(self,X):
        prob=self.predict_proba(X)
        return np.argmax(prob,axis=1)

def get_dataset_params(X,y):
    return {'n_cats':max(y)+1,'dims':X.shape[1],
        'batch_size':X.shape[0]}

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