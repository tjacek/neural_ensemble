import tools
tools.silence_warnings()
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score#,f1_score
from scipy import stats
import json

def single_exp(pred_path,out_path):
    acc_dict=metric_dict(accuracy_score,pred_path)
    df=make_df(acc_dict)
    df=df.sort_values(by=['mean'], ascending=False)
    print(df)
#    for data_i in df.dataset.unique():
#        pvalue_df=get_pvalue(data_i,df,acc_dict)
#        summary(pvalue_df, stats_type=1)

def summary(df_i, stats_type=1):
    if(stats_type==1):
        print(df_i[ (df_i.improv==True) &
                    (df_i.pvalue<0.05)])
    elif(stats_type==2):
        print(df_i[ (df_i.improv==True) &
              (df_i.pvalue>0.05)])
    else:
        print(df_i[ (df_i.improv==False) &
                    (df_i.pvalue<0.05)])

def make_df(acc_dict):
    lines=[]
    stats=[np.mean,np.std]
    for id_i,metric_i in acc_dict.items():
        line_i=id_i.split(',')
        line_i+=[round(stat_j(metric_i),4) 
                        for stat_j in stats]
        lines.append(line_i)
    cols=['dataset','clf','mean','std']
    return pd.DataFrame(lines,columns=cols)

def metric_dict(metric_i,pred_path):
    metric_dict={}
    for path_i in tools.top_files(pred_path):
        all_pred=read_pred(path_i)
        id_i=get_id(path_i)
        line_i=[get_id(path_i)]
        acc=[ metric_i(test_i,pred_i) 
                for test_i,pred_i in all_pred]
        metric_dict[id_i]=acc
    return metric_dict

def read_pred(path_i):
    with open(path_i, 'r') as f:        
        json_bytes = f.read()                      
        return json.loads(json_bytes)

def get_id(path_i):
    raw=path_i.split('/')
    return f'{raw[-2]},{raw[-1]}'

def get_pvalue(dataset,df,acc_dict):
    single,ens=[],[]
    for id_i in acc_dict:
        if('ens' in id_i):
            ens.append(id_i)
        else:
            single.append(id_i)
    single_mean=get_mean_dict(single,df,dataset)
    ens_mean=get_mean_dict(ens,df,dataset)
    lines=[]
    for single_i in single:
        for ens_j in ens:
            r=stats.ttest_ind(acc_dict[single_i], 
                acc_dict[ens_j], equal_var=False)
            p_value=round(r[1],4)
            single_clf_i=single_i.split(',')[-1]
            ens_clf_j=ens_j.split(',')[-1]
            diff_ij= ens_mean[ens_clf_j]-single_mean[single_clf_i]
            line_i=[dataset,single_clf_i,ens_clf_j]
            line_i+=[p_value,(diff_ij>0),diff_ij]
            lines.append(line_i)
    cols=['dataset','single','ens','pvalue','improv','diff']
    return pd.DataFrame(lines,columns=cols)

def get_mean_dict(single,df,dataset):
    single_acc={}
    for single_i in single:
        clf_i=single_i.split(',')[-1]
        df_i= df[(df.dataset==dataset) & (df.clf==clf_i)]
        single_acc[clf_i]=list(df_i['mean'])[0]
    return single_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='10_10/pred')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    if(args.dir>0):
        single_exp=tools.dir_fun(2)(single_exp)
    single_exp(args.pred,'out')