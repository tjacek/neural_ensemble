import numpy as np
import pandas as pd 
from sklearn import linear_model
import plot

def reg_frame(result,stats):
    if(type(result)==str):
        result=pd.read_csv(result)
    if(type(stats)==str):
        stats=pd.read_csv(stats)
    new_data=[]
    for data_i,row_i in plot.best_gen(result):
        common=result[ (result.dataset==data_i) & 
                       (result.ens=='common') &
                       (result.clf==row_i['clf'])]      
        diff_i=float(row_i['mean_acc'] - common['mean_acc'])
        stats_i= stats[stats.dataset==data_i]
        stats_i=stats_i.iloc[0].tolist()
        stats_i.append(diff_i)
        new_data.append(stats_i)
    new_cols=stats.columns.tolist()+['diff']
    new_data=pd.DataFrame(new_data, columns=new_cols)    
    return new_data

def linear_reg(df,robust=False):
    if(robust):
        clf=linear_model.HuberRegressor()
    else:
        clf=linear_model.LinearRegression()
    X=df[['classes','samples','features','gini']].to_numpy()
    y=df['diff'].to_numpy()
    clf.fit(X,y)
    return clf.coef_/np.sum(np.abs(clf.coef_))

result_path='../uci_bayes/bayes/result.csv'
stats_path='stats.csv'
df_reg=reg_frame(result_path,stats_path)
coff=linear_reg(df_reg)
print(coff)
coff=linear_reg(df_reg,True)
print(coff)
plot.scatter_plot(df_reg,x_col='features',y_col='diff')