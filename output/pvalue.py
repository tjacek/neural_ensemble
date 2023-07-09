import tools
import numpy as np
from scipy import stats

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
    return ','.join([best,worse,f'{pvalue:.4f}'])

pred_path='pred'
result_path="result.csv"
result_df=pd.read_csv(result_path)
for data_i in result_df['dataset'].unique():
	df_i=result_df[result_df['dataset']==data_i]
	acc_i=tools.metric_dict(f'{pred_path}/{data_i}','acc')
	acc_i={ key_i.split(',')[-1]:value_i 
	          for key_i,value_i in acc_i.items()}
	line_i=get_pvalue('RF',acc_i)
	print(f'\\hline {data_i},{line_i} \\\\')
#print(acc_dict.keys())