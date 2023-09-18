import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
import exp

if __name__ == '__main__':
    dir_path='../../pred'
    taboo=[[],
           ['binary',
            'multi'],
           ['SVC','LR'],
           ['cs','inliner']]
    acc_dict=exp.read_acc(dir_path)
    df=acc_dict.to_df(taboo)
    pvalue_df=list(exp.pvalues(df,acc_dict))
    stats_pd=pd.read_csv('stats.csv')
    
    df=pd.concat(pvalue_df)
    df['delta']=df['diff']
    df['sig_better']=df.apply(lambda x: int(x.sig>0 and x.delta>0), 
    	                      axis=1)
    df=df[['dataset','sig_better']]
    df= stats_pd.merge(df, on='dataset')
    X=df[['classes','samples','features','gini']].to_numpy()
    y=df['sig_better'].to_numpy()
    
    clf=LogisticRegression(solver='liblinear')#,
#            class_weight='balanced')
    clf.fit(X,y)
    print(clf.coef_ /np.sum(np.abs(clf.coef_)))