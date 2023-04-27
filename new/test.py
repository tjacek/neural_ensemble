import tools
tools.silence_warnings()
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from time import time
import clfs,learn,models,tools

def single_exp(data_path,model_path):
    df=pd.read_csv(data_path) 
    X,y=tools.prepare_data(df)
    model_io=models.ModelIO(model_path)
    all_models,splits=[],[]
    for clf_i,(train_i,test_i) in model_io.read():
        all_models.append(clf_i)     
        split_i= (X[train_i],y[train_i]),(X[test_i],y[test_i])  
        splits.append(split_i)
    eval('SVC', splits)
    eval('RF', splits)
    eval(all_models,splits)

def eval(clf, splits):
    split_iter,clf_str=get_iter(clf,splits)
    start=time()
    acc,balanced=[],[]
    for clf_i,split_i in split_iter():
        X_test,y_test=split_i
        y_pred=clf_i.predict(X_test)
        acc_i=accuracy_score(y_test,y_pred)
        balan_i=balanced_accuracy_score(y_test,y_pred)
        acc.append(acc_i)
        balanced.append(balan_i)
    end=time()
    print(f'Eval time-{clf_str}:{(end-start):.2f}s')
    print(f'Mean acc:{np.mean(acc):.4f},Std acc:{np.std(acc):.4f}')
    print(f'Mean balanced acc:{np.mean(balanced):.4f},Std acc:{np.std(balanced):.4f}')

def get_iter(clf,splits):
    if(type(clf)==list):
        def helper():
            for clf_i,(train_i,test_i) in zip(clf,splits):
                if(clfs.is_cpu(clf_i)):
#                    clf_i.multi_clf='LR-imb'
                    clf_i.train_clfs(train_i)	
                yield clf_i,test_i
        clf_str=str(clf[0])
    else:
        def helper():
            for train_i,test_i in splits:
                clf_i=learn.get_clf(clf)
                clf_i.fit(*train_i)
                yield clf_i,test_i
        clf_str=str(clf)
    return helper,clf_str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='uci/wine-quality-red')
    parser.add_argument("--models", type=str, default='out')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()
    single_exp(args.data,args.models)