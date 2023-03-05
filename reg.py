import numpy as np
import pandas as pd 
from sklearn import linear_model
import plot

def reg_dict(result_path,stats_path):
    result=pd.read_csv(result_path)
    stats=pd.read_csv(stats_path)
    X,y,names=[],[],[]
    for data_i,row_i in plot.best_gen(result):
        common=result[ (result.dataset==data_i) & 
                       (result.ens=='common') &
                       (result.clf==row_i['clf'])]      
        diff_i=(row_i['mean_acc'] - common['mean_acc'])
        y.append(diff_i)
        stats_i =stats[stats.dataset==data_i].iloc[0]
        X.append(stats_i.tolist()[1:])
    X,y=np.array(X),np.array(y)
    coef= lsm_coff(X,y)
    print(coef)

def lsm_coff(X,y):
    clf = linear_model.LinearRegression()#Lasso(alpha=0.001)
    clf.fit(X,y)
    return clf.coef_/np.sum(np.abs(clf.coef_))

result_path='../uci_bayes/bayes/result.csv'
stats_path='stats.csv'
reg_dict(result_path,stats_path)