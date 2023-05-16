import tools
tools.silence_warnings()
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import pred

def single_exp(pred_path,result_path,pvalue_path):
    pred_dict= pred.read_preds(pred_path)
    get_results(pred_dict,result_path)
    get_pvalue(pred_dict,pvalue_path,metric='balanced_acc')

def select_exp(pred_path,acc_path):
    pred_dict= pred.read_preds(pred_path)
    selected_dict = pred_dict.select(acc_path)
    get_results(selected_dict)
    get_pvalue(selected_dict)

def get_results(pred_dict,result_path=None):
    stats_dict= defaultdict(lambda :[])
    metrics=['acc','balanced_acc']#,'f1','recall','precision']
    for metric_i in metrics:
        metrict_dict=pred_dict.compute_metric(metric_i)
        for name_j,value_j in metrict_dict.items():
            stats_dict[name_j].append(np.mean(value_j))
            stats_dict[name_j].append(np.std(value_j))
    cols=['dataset']
    cols+=[f'{metric_i}_{stats}'  
            for metric_i in metrics
                for stats in ['mean','std']]
    lines=[ [name_i]+value_j 
        for name_i,value_j in stats_dict.items()]
    df= pd.DataFrame(lines,columns=cols)
    print(df)
    if(not (result_path is None)):
        df.to_csv(result_path, index=False)

def get_pvalue(pred_dict,pvalue_path=None,metric='balanced_acc'):
    metric_dict=pred_dict.compute_metric(metric)
    ens_names,clf_names=[],[]
    for key_i in metric_dict:
        if('NECSCF' in key_i):
            ens_names.append(key_i)
        else:
            clf_names.append(key_i)
    lines=[]
    for ens_i in ens_names:
        for clf_j in clf_names:
            r=stats.ttest_ind(metric_dict[ens_i], 
                metric_dict[clf_j], equal_var=False)
            p_value=round(r[1],4)
            line_ij=[ens_i,clf_j,p_value,(p_value<0.05)]
            lines.append(line_ij)
    df=pd.DataFrame(lines,columns=['ens','clf','p_value','sig'])
    print(df)
    if(not (pvalue_path is None)):
        df.to_csv(pvalue_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='lymphography/pred')
    parser.add_argument("--results", type=str, default='lymphography/acc.txt')
    parser.add_argument("--p_value", type=str, default='vehicle/p_value')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()
    single_exp(args.pred,args.results,args.p_value)
#    select_exp(args.pred,args.results)