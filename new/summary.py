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

        print(full_df)
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

#def best(dir_path,metric='balanced_acc_mean'):
#    paths=tools.get_dirs(dir_path)
#    cols=['dataset','best','second','p_value','sig']
#    lines=[]
#    for path_i in paths:
#        name_i=path_i.split('/')[-1]
#        result_i=f'{path_i}/results'
#        df=pd.read_csv(result_i) 
#        df['ens']=df['clf'].apply(lambda clf_i: ('NECSCF' in clf_i))
#        df_pvalue=pd.read_csv(f'{path_i}/pvalue.txt') 
#        df=df.sort_values(by=metric,ascending=False)

#        best,is_ens=df.iloc[0]['clf'],df.iloc[0]['ens']
        
#        tmp= df[df['ens']==(not is_ens)].sort_values(by=metric,ascending=False)
#        second= tmp.iloc[0]['clf']

#        ens_type,clf_type = (best,second)  if(is_ens) else (second,best)

#        p_row=df_pvalue[ (df_pvalue['ens']==ens_type) &
#                    (df_pvalue['clf']==clf_type)]
#        lines.append([name_i,best,second,float(p_row['p_value']),str(p_row['sig'])])
#    df= pd.DataFrame(lines,columns=cols)
#    print(df)

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