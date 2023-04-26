import tools
tools.silence_warnings()
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
import models,tools,learn
from collections import defaultdict

def single_exp(data_path,model_path,result_path):
    df=pd.read_csv(data_path) 
    X,y=tools.prepare_data(df)
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
        y_pred=model_i.predict(X_test)
        acc_dict['NECSCF'].append((y_test,y_pred))#accuracy_score(y_test,y_pred))
    metrics=[accuracy_score,balanced_accuracy_score]#,f1_score]
    cols=['clf','acc_mean','acc_std','balan_acc_mean','balan_acc_std']
    lines=[]
    for name_i,result_i in acc_dict.items():
        line_j=[name_i]
        for metric_j in metrics:
            stats_i=[metric_j(y_true,y_pred) for y_true,y_pred in result_i]
            line_j+=[np.mean(stats_i),np.std(stats_i)]
        lines.append(line_j)	
    df= pd.DataFrame(lines,columns=cols)
    df.to_csv(result_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='uci/cmc')
    parser.add_argument("--models", type=str, default='uci_out/models/cmc')
    parser.add_argument("--results", type=str, default='uci_out/results/cmc')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()
    single_exp(args.data,args.models,args.results)