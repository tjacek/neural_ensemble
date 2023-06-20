import tools
tools.silence_warnings()
from tensorflow.keras import Input, Model
import keras_tuner as kt
import argparse
import data

class MultiKTBuilder(object): 
    def __init__(self,params):#,hidden=[(0.25,5),(0.25,5)]):
        self.params=params
        self.hidden=hidden

    def __call__(self,hp):
        model = tf.keras.Sequential()
        for i in range(hp.Int('layers', 1, 2)):
            hidden_i=hp.Int('units_' + str(i), 
                    min_value= int(self.params['dims']*self.hidden[0]), 
                    max_value=int(self.params['dims']*self.hidden[1]), 
                    step=10)
            model.add(tf.keras.layers.Dense(units=hidden_i ))
#                                    activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid'])))
        model.add(tf.keras.layers.Dense(self.params['n_cats'], activation='softmax'))
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

#    def get_params_names(self):
#        params=[f'n_hidden{j}' for j in range(len(self.hidden))]              

#    def layer_name(self,i,j):
#        if(j==(len(self.hidden)-1)):
#            return f'hidden{i}'
#        return f'{i}_{j}'

def single_exp(data_path,hyper_path,n_split,n_iter):
    X,y=data.get_dataset(data_path)
    data_params=data.get_dataset_params(X,y)
    print(data_params)
#    best=bayes_optim(X,y,data_params,n_split,n_iter)
#    print(best)
#    with open(hyper_path,"a") as f:
#        f.write(f'{str(best)}\n') 
#    return best

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='data/wine-quality-red')
    parser.add_argument("--hyper", type=str, default='hyper.txt')
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=2)
    args = parser.parse_args()
    single_exp(args.data,args.hyper,args.n_split,args.n_iter)