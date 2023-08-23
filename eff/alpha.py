import tools
tools.silence_warnings()
import os,argparse
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import data,deep,ens

class AlphaEns(deep.NeuralEnsemble):
    def __init__(self, model,params,hyper_params,split):#,alpha_values):
        super().__init__(model,params,hyper_params,split)
#        self.alpha_values=alpha_values
        
    def get_type(self):
        return 'alpha'

    def fit(self,x,y,batch_size,epochs=150,verbose=0,callbacks=None):
        X,y=self.split.get_all(x,y,train=True)
        y=[ tf.keras.utils.to_categorical(y_i) 
              for k in range(self.params['n_cats'])
                  for y_i in y]
        self.model.fit(x=X,
                       y=y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=callbacks)

def build_alpha(params,hyper_params,split):
    model=ens.ens_builder(params,
                          hyper_params,
                          n_splits=len(split))
    metrics={f'output_{i}_{k}':'accuracy' 
                for i in range(len(split))
                    for k in range(params['n_cats'])}
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=metrics)
    deep_ens=AlphaEns(model=model,
                      params=params,
                      hyper_params=hyper_params,
                      split=split)
    return deep_ens


def alpha_exp(data_path,hyper_path,n_splits=3):
    dataset= data.get_dataset(data_path)
    hyper_df=pd.read_csv(hyper_path)
    splits=dataset.get_splits(n_splits,1)
#    cv = RepeatedStratifiedKFold(n_splits=n_splits, 
#                                 n_repeats=1,#n_repeats, 
#                                 random_state=4)


if __name__ == '__main__':
    dir_path='../../s_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../s_uci/cleveland')
    parser.add_argument("--hyper", type=str, default=f'{dir_path}/hyper.csv')    
    parser.add_argument("--n_splits", type=int, default=3)
#    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--log", type=str, default=f'{dir_path}/log.info')
    args = parser.parse_args()
#    tools.start_log(args.log)
    alpha_exp(args.data,args.hyper,args.n_splits)