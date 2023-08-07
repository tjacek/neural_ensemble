import tools
tools.silence_warnings()
import pandas as pd
from keras import callbacks
import argparse
import data,deep,ens,learn,tools

def train_exp(data_path,hyper_path,model_path,n_splits=10,n_repeats=10):
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
    tools.make_dir(model_path)
    for i,split_i in enumerate(all_splits):
        deep_ens=ens.build_multi(params=params,
                                     hyper_params=hyper_dict,
                                     split=split_i)
        deep_ens.fit(x=dataset.X,
                     y=dataset.y,
                     epochs=150,
                     batch_size=params['batch'],
                     verbose=1,
                     callbacks=early_stop)
        deep_ens.save(f'{model_path}/{i}')

def pred_exp(data_path,hyper_path,model_path):
    dataset=data.get_dataset(data_path)
    accuracy=tools.get_metric('acc')
    for model_path_i in tools.top_files(model_path):
        deep_ens=ens.read_ens(model_path_i)
        y_pred=deep_ens.predict_classes(dataset.X)
        acc_i=accuracy(dataset.y,y_pred)
        print(acc_i) 
        deep_ens.extract(dataset.X)


def extract_exp(data_path,hyper_path,model_path):
    dataset=data.get_dataset(data_path)
    accuracy=tools.get_metric('acc')
    for model_path_i in tools.top_files(model_path):
        deep_ens=ens.read_ens(model_path_i)
        cs_feats_i=deep_ens.extract(dataset.X)
        necscf=learn.NECSCF(dataset=dataset,
                            split=deep_ens.split,
                            cs_feats=cs_feats_i)
        necscf('LR')
#        y_pred=deep_ens.predict_classes(dataset.X)
#        acc_i=accuracy(dataset.y,y_pred)
#        print(acc_i) 
#        deep_ens.extract(dataset.X)

if __name__ == '__main__':
    dir_path='../../optim_alpha/s_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../s_uci/cleveland')
    parser.add_argument("--hyper", type=str, default=f'{dir_path}/hyper.csv')
    parser.add_argument("--models", type=str, default=f'models2')
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=10)
    args = parser.parse_args()
    extract_exp(data_path=args.data,
                hyper_path=args.hyper,
                model_path=args.models)#,
#              n_splits=args.n_splits,
#              n_repeats=args.n_repeats)