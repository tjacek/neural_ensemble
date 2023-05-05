import tools
tools.silence_warnings()
import argparse
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from scipy import stats
from collections import defaultdict,OrderedDict
import clfs,models,learn

def single_exp(data_path,model_path,result_path,p_value):
    df=pd.read_csv(data_path) 
    X,y=tools.prepare_data(df)
    acc_dict=get_pred_dict(X,y,model_path)
    get_results(acc_dict,result_path)
#    get_pvalue(acc_dict,p_value)

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
            line_j+=[ round(fun_k(stats_i),4)
                for fun_k in [np.mean,np.std]]
        lines.append(line_j)
    df= pd.DataFrame(lines,columns=cols)
    print(df)
    df.to_csv(result_path, index=False)

def get_pred_dict(X,y,model_path):
    clf=['RF','SVC']
    model_reader=models.ManyClfs(model_path)
    pred_dict=defaultdict(lambda :[])
    for model_i,(train_i,test_i) in model_reader.read():
        X_train,y_train=X[train_i],y[train_i]
        X_test,y_test=X[test_i],y[test_i]
        for name_j,clf_j in iter_clfs((X_train,y_train),clf,model_i):
            y_pred=clf_j.predict(X_test)
            pred_dict[name_j].append((y_test,y_pred))
    return pred_dict

def iter_clfs(train_i,clf_types,model_dict):
    for clf_type_j in clf_types:
        clf_j=learn.get_clf(clf_type_j)
        clf_j.fit(*train_i)
        yield clf_type_j,clf_j
    for clf_type_j,clf_j in model_dict.items():
        if(clfs.is_cpu(clf_j)):
            for clf_type_k in clf_types:
                clf_j.train_clfs(train_i,clf_type_k)
                yield f'NECSCF({clf_type_k})',clf_j
        else:
            yield 'NECSCF(NN-TF)',clf_j

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