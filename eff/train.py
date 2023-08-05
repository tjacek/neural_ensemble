import tools
tools.silence_warnings()
import pandas as pd
from keras import callbacks
import argparse
import data,deep,tools

def train_exp(data_path,hyper_path,out_path,n_splits=10,n_repeats=10):
    dataset=data.get_dataset(data_path)
    all_splits=dataset.get_splits(n_splits=n_splits,
                                  n_repeats=n_repeats)
    params=dataset.get_params()
    hyper_df=pd.read_csv(hyper_path)
    name_i=data_path.split('/')[-1]
    hyper_dict=tools.get_hyper(name_i,hyper_df)
    early_stop = callbacks.EarlyStopping(monitor='accuracy',
                                         mode="max", 
                                         patience=5,
                                         restore_best_weights=True)
    tools.make_dir(out_path)
    for i,split_i in enumerate(all_splits):
        deep_ens=deep.build_ensemble(params=params,
                                     hyper_params=hyper_dict,
                                     split=split_i)
        deep_ens.fit(x=dataset.X,
                     y=dataset.y,
                     epochs=150,
                     batch_size=params['batch'],
                     verbose=1,
                     callbacks=early_stop)
        deep_ens.save(f'{out_path}/{i}')

if __name__ == '__main__':
    dir_path='../../optim_alpha/s_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../s_uci/cleveland')
    parser.add_argument("--hyper", type=str, default=f'{dir_path}/hyper.csv')
    parser.add_argument("--models", type=str, default=f'models')
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=10)
    args = parser.parse_args()
    train_exp(data_path=args.data,
              hyper_path=args.hyper,
              out_path=args.models,
              n_splits=args.n_splits,
              n_repeats=args.n_repeats)