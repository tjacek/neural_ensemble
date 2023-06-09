import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import tools,pred

def make_summary(in_path,dir_path,out_path,metric='acc_mean'):
    for name_i,df,df_pvalue in result_iter(in_path,dir_path,metric):
        with open(out_path,"a") as f:
            f.write(f'{name_i}\n')
            f.write(df.to_csv())
            f.write(df_pvalue.to_csv())

def result_iter(in_path,dir_path,metric='acc_mean'):
    for path_i in tools.get_dirs(in_path):
        name_i=path_i.split('/')[-1]
        dir_i=f'{path_i}/{dir_path}'
        df=pd.read_csv(f'{dir_i}/results')
        df=df.sort_values(by=metric,ascending=False)
        df_pvalue=pd.read_csv(f'{dir_i}/pvalue.txt')
        yield name_i,df,df_pvalue

def comp_summary(in_path:str,dirs:list,out_path:str):
    df_dict=make_dataframes(in_path,dirs)
    for name_i,df_list in df_dict.items():
        full_df=pd.concat(df_list)
        full_df=full_df.sort_values(by='acc_mean',ascending=False)
        csv_i=full_df.to_csv()
        to_latex(csv_i)
#        with open(out_path,"a") as f:
#            f.write(f'{name_i}\n')
#            for df_i in df_list:
#                f.write(df_i.to_csv())

def make_dataframes(in_path,dirs):
    df_dict=defaultdict(lambda:[])
    for dir_i in dirs:
        for name_j,df,df_pvalue in  result_iter(in_path,dir_i):
            df['clf']=df['dataset'].apply(lambda alg_i: get_clf(alg_i))
            df['variant']=df['dataset'].apply(lambda alg_i: get_variant(alg_i))
            df['variant']=df['variant'].apply(rename())
            df['dataset']=df['dataset'].apply(lambda alg_i: name_j)
            df=df.round(decimals=4)
            df=df[['dataset','variant','clf','acc_mean','acc_std']]
            df_dict[name_j].append(df)
    return df_dict

def get_clf(name_i):
    clf_i=name_i.split('(')[1]
    clf_i=clf_i.replace(")","")
    return clf_i

def get_variant(name_i):
    clf_i=name_i.split('(')[0]
    clf_i=clf_i.replace(")","")
    return clf_i

def rename():
    var_dict={'better':'selection','NECSCF2':'all_clfs'}
    def helper(variant_i):
        if(not variant_i in var_dict):
            return variant_i
        return var_dict[variant_i]
    return helper

def to_latex(csv_i):
    for line_i in csv_i.split('\n'):
        raw_i=' & '.join( line_i.split(','))
        print('\hline' + raw_i + '\\\\')

#~!@#$%^&*()_+   QWERT&*//
def acc_summary(dir_path):
    for path_i in tools.get_dirs(dir_path):
        print(path_i)
        acc_path_i=f'{path_i}/acc.txt'
        acc_dict=pred.read_acc_dict(acc_path_i)
        for j,clf_j in acc_dict.items():
            acc_j= list(clf_j.values())
            stats_j=[f'{fun(acc_j):.2f}' for fun in [np.mean,np.median,np.max,np.min]]
            print(','.join(stats_j))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default='../../mult_acc')
    parser.add_argument("--dir", type=str, default='high')
    parser.add_argument("--out_path", type=str, default='../../mult_acc/comp.txt')#/summary.txt')

    args = parser.parse_args()
    print(args)
    out_path=f'{args.out_path}/{args.dir}'
#    make_summary(args.in_path,args.dir,out_path)
    comp_summary(args.in_path,['raw2','better','best2'],args.out_path)