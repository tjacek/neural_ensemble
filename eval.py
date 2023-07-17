import tools
tools.silence_warnings()
import argparse
import numpy as np
import pandas as pd
#from sklearn.metrics import accuracy_score,balanced_accuracy_score#,f1_score
from scipy import stats
from  summary.metric_dict import make_acc_dict
import json

def single_exp(pred_path,out_path):
    acc_dict=make_acc_dict(pred_path,'acc')
    df_dict=summary(acc_dict)
    return list(df_dict.values())[0]

def sig_stats(df_dict):
    clfs=list(df_dict.values())[0]['clf'].unique()
    counter_dict={clf_i:[0,0,0,0] for clf_i in clfs }
    for data_i,df_i in df_dict.items():
        for j,row_j in df_i[df_i['diff']!=0].iterrows():
            clf_j,sig_j,diff_j=row_j['clf'],row_j['sig'],row_j['diff']
            if(diff_j<0):
                diff_j=0
            index=int(sig_j)*2+ int(diff_j)
            counter_dict[clf_j][index]+=1
    print(counter_dict)

def summary(acc_dict):
    data_dict=acc_dict.dataset_dfs()
    df_dict={}
    for data,df_i in data_dict.items():
        lines=[]
        variant_df=df_i[df_i['variant']=='-']
        for index, row in variant_df.iterrows():
            lines.append(row.tolist())
        for clf_j in df_i['clf'].unique():
            df_j=df_i[(df_i['clf']==clf_j) &
                       (df_i['variant']!='-')]
            df_j=df_j.sort_values(by='mean',ascending=False)
            line_j=df_j.iloc[0].tolist()
            lines.append(line_j)
        cols=['dataset','id', 'mean','std','clf','cs','alpha','variant']
        s_df=pd.DataFrame(lines,columns=cols)
        s_df=s_df.sort_values(by='mean',ascending=False)
        s_df=add_pvalues(data,s_df,acc_dict)
        s_df=show_impr(s_df)
        df_dict[data]=s_df
    return df_dict

def add_pvalues(data,s_df,acc_dict,verbose=False):
    base_cls= s_df[s_df['variant']=='-']
    id_dict={  row_i['clf']:row_i['id'] 
             for i,row_i in base_cls.iterrows()}
    def helper(x):
        if(x.variant=='-'):
            return 1.0
        if(not x.clf in id_dict):
            return '-'
        base_id=f'{data},{id_dict[x.clf]}'
        x_id=f'{data},{x.id}'
        return  acc_dict.pvalue(base_id,x_id)
    s_df['pvalue']=s_df.apply( helper,axis=1)
    s_df.drop('id', inplace=True, axis=1)
    s_df=s_df[s_df['pvalue']!='-']
    s_df['sig']=s_df['pvalue'].apply(lambda p: p<0.05)
    if(verbose):
        print(s_df)
    return s_df

def show_impr(s_df):
    base_cls= s_df[s_df['variant']=='-']
    mean_dict={  row_i['clf']:row_i['mean'] 
             for i,row_i in base_cls.iterrows()}
    def helper(x):
        if(x.variant=='-'):
            return 0
        return np.sign(x['mean']-mean_dict[x.clf] )
    s_df['diff']=s_df.apply( helper,axis=1)
    return s_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='../s_10_10/pred')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    if(args.dir>0):
        single_exp=tools.dir_fun(2)(single_exp)
    df_dict=single_exp(args.pred,'out')
    sig_stats(df_dict)