import numpy as np
import pandas as pd 
from sklearn import linear_model
from scipy import stats
import statsmodels.api as sma
from sklearn import preprocessing

import plot

def reg_frame(result,stats,clf=None):
    if(type(result)==str):
        result=pd.read_csv(result)
    if(type(stats)==str):
        stats=pd.read_csv(stats)
    for col_i in stats.columns:
        def helper(data_j):
            index_j=(stats['dataset']==data_j)
            return stats[index_j][col_i].to_list()[0]
        result[col_i]=result['dataset'].apply(helper)
    return result

def linear_reg(df,robust=False):
    if(robust):
        clf=linear_model.HuberRegressor()
    else:
        clf=linear_model.LinearRegression()
    X=df[['classes','samples','features','gini']].to_numpy()
    y=df['diff'].to_numpy()
    clf.fit(X,y)
    return clf.coef_/np.sum(np.abs(clf.coef_))
    

def p_value(df):
    X=df[['classes','samples','features','gini']].to_numpy()
    y=df['diff'].to_numpy()
    est  = sma.OLS(y, X)
    output  = est.fit()
    print(f'pvalue:{output.f_pvalue}')
    print(output.summary2().tables[1]['P>|t|'])



result_path='diff.csv'
stats_path='stats.csv'
df_reg=reg_frame(result_path,stats_path)
coff=linear_reg(df_reg)
print([f'{c:2f}' for c in coff])
p_value(df_reg)