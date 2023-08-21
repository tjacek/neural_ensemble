import tools
tools.silence_warnings()
import pandas as pd
from keras import callbacks
import os,argparse
import data,deep,ens,learn,tools

def train_exp(data_path,hyper_path,model_path,n_splits=10,n_repeats=10):
    hyper_df=pd.read_csv(hyper_path)
    early_stop = callbacks.EarlyStopping(monitor='accuracy',
                                         mode="max", 
                                         patience=5,
                                         restore_best_weights=True)
    builder_fun={'base':deep.make_base,
                 'weighted':ens.WeightedBuilder(0.5),
                 'binary':ens.BinaryBuilder(0.5),
                 'multi': ens.build_multi}
    tools.make_dir(model_path)
    for ens_type,builder in builder_fun.items():
        @tools.log_time(task=f'TRAIN-{ens_type}')
        def helper(data_path,model_path):
            dataset=data.get_dataset(data_path)
            all_splits=dataset.get_splits(n_splits=n_splits,
                                      n_repeats=n_repeats)
            params=dataset.get_params()
            name_i=data_path.split('/')[-1]
            print(name_i)
            hyper_dict=tools.get_hyper(name_i,hyper_df)
            tools.make_dir(model_path)
            for i,split_i in enumerate(all_splits):
                deep_ens=builder(params=params,
                                 hyper_params=hyper_dict,
                                 split=split_i)
                deep_ens.fit(x=dataset.X,
                         y=dataset.y,
                         epochs=150,
                         batch_size=params['batch'],
                         verbose=1,
                         callbacks=early_stop)
                deep_ens.save(f'{model_path}/{i}')
        if(os.path.isdir(data_path)):
            helper=tools.dir_fun(2)(helper)
        helper(data_path,f'{model_path}/{ens_type}')

if __name__ == '__main__':
    dir_path='../../s_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../s_uci/')
    parser.add_argument("--hyper", type=str, default=f'../../hyper.csv')
    parser.add_argument("--models", type=str, default=f'../../models')
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--log", type=str, default=f'log.info')
    args = parser.parse_args()
    tools.start_log(args.log)
    train_exp(data_path=args.data,
                hyper_path=args.hyper,
                model_path=args.models,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats)