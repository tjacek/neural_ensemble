import pandas as pd 
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
    df['sig_better']=df.apply(lambda x: (x.sig>0 and x.delta>0), 
    	                      axis=1)
    df=df[['dataset','sig_better']]
    print(df)