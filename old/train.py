import tools
tools.silence_warnings()
import argparse
import os
import pandas as pd
import data,exp

def train_exp(data_path,hyper_path,out_path,n_splits=10,n_repeats=10):
    hyper_df=pd.read_csv(hyper_path)
    algs=['base','multi','binary','weighted']
    @tools.log_time(task='TRAIN')
    def helper(data_i,out_path):
        print(data_i)
        X,y=data.get_dataset(data_i)
        name_i=data_i.split('/')[-1]
        hyper_i=get_hyper(name_i,hyper_df)
        dataset_params=data.get_dataset_params(X,y)
        exp_factory=exp.ExpFactory(dataset_params,hyper_i)
        all_splits=data.gen_splits(X,y,
    	                           n_splits=n_splits,
    	                           n_repeats=n_repeats) 
        tools.make_dir(out_path)
        for alg_i in algs:
            tools.make_dir(f'{out_path}/{alg_i}')
            alg_params=get_alg_params(alg_i,hyper_i)
#            print(alg_params)
            for j,split_j in enumerate(all_splits.splits):
                 out_j=f'{out_path}/{alg_i}/{j}'
                 exp_j= exp_factory(X,y,split_j,alg_params)
                 exp_j.save(out_j)
    if(os.path.isdir(data_path)):
        helper=tools.dir_fun(2)(helper)
    helper(data_path,out_path)

def get_alg_params(name_i,hyper_i):
    if(name_i=='binary' or  name_i=='weighted'):
        return name_i,hyper_i['alpha']
    return name_i

def get_hyper(name_i,hyper_df):
    hyper_i=hyper_df[hyper_df['dataset']==name_i]
    hyper_i= hyper_i.iloc[0].to_dict()
    layers= [key_i for key_i in hyper_i
                   if('unit' in key_i)]
    layers.sort()
    hyper_i['layers']=[hyper_i[name_j] 
                          for name_j in layers]
    return hyper_i

if __name__ == '__main__':
    dir_path='../optim_alpha/s_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../s_uci')
    parser.add_argument("--hyper", type=str, default=f'{dir_path}/hyper.csv')
    parser.add_argument("--models", type=str, default=f'{dir_path}/models')
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=10)
#    parser.add_argument("--dir", type=int, default=0)
    parser.add_argument("--log", type=str, default=f'{dir_path}/log.info')
    args = parser.parse_args()
    tools.start_log(args.log)
    print(args)
#    if(args.dir>0):
#        single_exp=tools.dir_fun(3)(single_exp)
    train_exp(args.data,args.hyper,args.models,args.n_splits,args.n_repeats)