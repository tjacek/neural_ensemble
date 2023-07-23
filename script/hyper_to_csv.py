import argparse
import os
from collections import defaultdict
import pandas as pd

def to_csv(hyper_path,alpha_path,csv_path):
    alpha_df= pd.read_csv(alpha_path)
#    alpha_df.columns=['dataset','alpha','acc']
    hyper_dict= defaultdict(lambda :[])
    for data_i in os.listdir(hyper_path):
        hyper_dict['dataset'].append(data_i)
        with open(f'{hyper_path}/{data_i}') as f:
            hyper_i = eval(f.readlines()[-1])[0]
            for name_j,value_j in hyper_i.items():
                hyper_dict[name_j].append(value_j)
        df_i=alpha_df[ alpha_df['dataset']==data_i ]
        hyper_dict['alpha'].append(df_i['alpha'].to_list()[0])	
    hyper_df= pd.DataFrame(hyper_dict)
    print(hyper_df)
    hyper_df.to_csv(csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper", type=str, default='../../s_10_10/hyper')
    parser.add_argument("--alpha", type=str, default='../s_alpha.csv')
    parser.add_argument("--csv", type=str, default='hyper.csv')
    args = parser.parse_args()
    to_csv(args.hyper,args.alpha,args.csv)