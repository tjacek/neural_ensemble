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
#    df=acc_dict.to_df()
#    df=df.sort_values(by=['dataset','mean'], ascending=False)
#    print(df)
#    data_dict=acc_dict.dataset_dfs()
    summary(acc_dict)
#    for data_i in df.dataset.unique():
#        pvalue_df=get_pvalue(data_i,df,acc_dict)
#        summary(pvalue_df, stats_type=1)

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
        df_dict[data]=show_pvalues(data,s_df,acc_dict)
    return df_dict

def show_pvalues(data,s_df,acc_dict):
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
        print(id_dict[x.clf])
        return  acc_dict.pvalue(base_id,x_id)
    s_df['pvalue']=s_df.apply( helper,axis=1)
    s_df.drop('id', inplace=True, axis=1)
    s_df=s_df[s_df['pvalue']!='-']
    s_df['sig']=s_df['pvalue'].apply(lambda p: p<0.05)
    print(s_df)
    return s_df
#def get_pvalue(dataset,df,acc_dict):
#    single,ens=[],[]
#    for id_i in acc_dict:
#        if('ens' in id_i):
#            ens.append(id_i)
#        else:
#            single.append(id_i)
#    single_mean=get_mean_dict(single,df,dataset)
#    ens_mean=get_mean_dict(ens,df,dataset)
#    lines=[]
#    for single_i in single:
#        for ens_j in ens:
#            r=stats.ttest_ind(acc_dict[single_i], 
#                acc_dict[ens_j], equal_var=False)
#            p_value=round(r[1],4)
#            single_clf_i=single_i.split(',')[-1]
#            ens_clf_j=ens_j.split(',')[-1]
#            diff_ij= ens_mean[ens_clf_j]-single_mean[single_clf_i]
#            line_i=[dataset,single_clf_i,ens_clf_j]
#            line_i+=[p_value,(diff_ij>0),diff_ij]
#            lines.append(line_i)
#    cols=['dataset','single','ens','pvalue','improv','diff']
#    return pd.DataFrame(lines,columns=cols)

def get_mean_dict(single,df,dataset):
    single_acc={}
    for single_i in single:
        clf_i=single_i.split(',')[-1]
        df_i= df[(df.dataset==dataset) & (df.clf==clf_i)]
        single_acc[clf_i]=list(df_i['mean'])[0]
    return single_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='../s_10_10/pred')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    if(args.dir>0):
        single_exp=tools.dir_fun(2)(single_exp)
    single_exp(args.pred,'out')