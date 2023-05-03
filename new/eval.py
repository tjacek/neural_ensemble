import tools
tools.silence_warnings()
import argparse
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
import clfs,models,learn
from scipy import stats
from collections import defaultdict,OrderedDict

def single_exp(data_path,model_path,result_path,p_value):
    df=pd.read_csv(data_path) 
    X,y=tools.prepare_data(df)
    acc_dict=get_acc_dict(X,y,model_path)
    get_results(acc_dict,result_path)
    get_pvalue(acc_dict,p_value)

def get_results(pred_dict,result_path):
    metrics_dict={'acc':accuracy_score,
                  'balanced_acc':balanced_accuracy_score,
                  'f1':learn.f1_metric,
                  'recall':learn.recall_metric,
                  'precision':learn.precision_metric}
    metrics_dict=OrderedDict(metrics_dict)
    cols=['clf']
    cols+=[ f'{key_i}_{stat_i}'
                for key_i in metrics_dict
                    for stat_i in ['mean','std'] ]
    lines=[]
    for name_i,result_i in pred_dict.items():
        line_j=[name_i]
        for metric_name_j,metric_j in metrics_dict.items():
            stats_i=[metric_j(*pred) for pred in result_i]
            line_j+=[np.mean(stats_i),np.std(stats_i)]
        lines.append(line_j)
    df= pd.DataFrame(lines,columns=cols)
    df.to_csv(result_path, index=False)

def get_acc_dict(X,y,model_path):
    clf=['RF','SVC']
    model_reader=models.ModelIO(model_path)
    acc_dict=defaultdict(lambda :[])
    for model_i,(train_i,test_i) in model_reader.read():
        X_train,y_train=X[train_i],y[train_i]
        X_test,y_test=X[test_i],y[test_i]
        for clf_type_j in clf:
            clf_j=learn.get_clf(clf_type_j)
            clf_j.fit(X_train,y_train)
            y_pred= clf_j.predict(X_test)
            acc_dict[clf_type_j].append((y_test,y_pred)) #accuracy_score(y_test,y_pred))
        if(clfs.is_cpu(model_i)):
            model_i.train_clfs((X_train,y_train))#,'LR-imb')
        y_pred=model_i.predict(X_test)
        acc_dict['NECSCF'].append((y_test,y_pred))
    return acc_dict

def get_pvalue(acc_dict,pvalue_path):
    keys=[ key_i for key_i in acc_dict.keys()
            if(key_i!='NECSCF')]
    metric=accuracy_score
    ens_i=[metric(*result_i) for result_i in acc_dict['NECSCF']]
    cols=['base clf','p_value','sig']
    lines=[]
    for key_i in keys:
        acc_i=[metric(*result_i) for result_i in acc_dict[key_i]]
        r=stats.ttest_ind(acc_i, ens_i, equal_var=False)
        p_value=r[1]
        line_i=[key_i,p_value,(p_value<0.05)]
        lines.append(line_i)
    df=pd.DataFrame(lines,columns=cols)
    print(df)
    df.to_csv(pvalue_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='uci/vehicle')
    parser.add_argument("--models", type=str, default='vehicle/models')
    parser.add_argument("--results", type=str, default='vehicle/results2')
    parser.add_argument("--p_value", type=str, default='vehicle/p_value')

    parser.add_argument("--log_path", type=str, default='vehicle/log.time')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()
    tools.start_log(args.log_path)
    start=time()
    single_exp(args.data,args.models,args.results,args.p_value)
    tools.log_time(f'EVAL:{args.data}',start) 