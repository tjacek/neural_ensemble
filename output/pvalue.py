import tools
import numpy as np
from scipy import stats
import re
import pandas as pd

def get_pvalue(clf_i,acc_i):
    pairs=[ (key_j,np.mean(value_j)) 
            for key_j,value_j in acc_i.items()
                if(clf_i in key_j and clf_i!=key_j)]
    names,mean=zip(*pairs)
    k=np.argmax(mean)
    best,best_value=names[k],mean[k]
    clf_value= np.mean(acc_i[clf_i])
    if(best_value>clf_value):
        best,worse=best,clf_i
    else:
        best,worse=clf_i,best
    pvalue=stats.ttest_ind(acc_i[best],acc_i[worse],
                equal_var=False)[1]
    return [best,worse,round(pvalue,4)]#f'{pvalue:.4f}']

def get_pvalue_frame(pred_path,result_path,clf='RF'):
    result_df=pd.read_csv(result_path)
    lines=[]
    for data_i in result_df['dataset'].unique():
        df_i=result_df[result_df['dataset']==data_i]
        acc_i=tools.metric_dict(f'{pred_path}/{data_i}','acc')
        acc_i={ key_i.split(',')[-1]:value_i 
              for key_i,value_i in acc_i.items()}
        line_i=[data_i]+get_pvalue(clf,acc_i)
        lines.append(line_i)
    cols=['dataset','better','worse','pvalue']    
    df = pd.DataFrame(lines,columns=cols)
    def helper(x):
        if(x==clf):
            return f'-/-/{clf}'
        x=x.replace('_ens-','/')
        x=x.replace('_ens','/-')
        x=x.replace('(','/')
        x=x.replace(')','')
        return x
    df['better']=df['better'].apply(helper)
    df['worse']=df['worse'].apply(helper)
    df=df.sort_values(by='pvalue')#,ascending=False)
    print(df.to_latex())



pred_path='pred'
result_path="result.csv"
get_pvalue_frame(pred_path,result_path,clf='SVC')
#	print(f'\\hline {data_i},{line_i} \\\\')
#print(acc_dict.keys())