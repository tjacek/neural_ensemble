import tools
tools.silence_warnings()
import argparse
import pandas as pd
import exp

def train_exp(data_path,hyper_path,out_path,n_splits=10,n_repeats=10):
    hyper_df=pd.read_csv(hyper_path)
    print(hyper_df)

if __name__ == '__main__':
    dir_path='../optim_alpha/s_3_3'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../s_uci')
    parser.add_argument("--hyper", type=str, default=f'{dir_path}/hyper.csv')
    parser.add_argument("--models", type=str, default=f'{dir_path}/models')
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--dir", type=int, default=0)
    parser.add_argument("--log", type=str, default=f'{dir_path}/log.info')
    args = parser.parse_args()
    tools.start_log(args.log)
    print(args)
#    if(args.dir>0):
#        single_exp=tools.dir_fun(3)(single_exp)
    train_exp(args.data,args.hyper,args.models,args.n_splits,args.n_repeats)