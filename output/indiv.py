import numpy as np
import pandas as pd
import re
import tools

def get_alpha(raw):
    digits=re.findall(r'\d+',raw)
    if(len(digits)>0):
        return f'0.{digits[1]}'
    return '-'

def ens_quality(in_path,clf_i):    
    lines=[]
    for data_i in tools.top_files(in_path):
        acc_i=tools.metric_dict(data_i,'acc',False)
        data_i= data_i.split('/')[-1]
        names=[key_j for key_j in acc_i.keys()
               if(clf_i in key_j and key_j!=clf_i)]
        clf_acc=acc_i[clf_i]
        def helper(name_k):
            comp_k=[ int(ens_t>clf_t)
                for clf_t,ens_t in zip(clf_acc,acc_i[name_k])]
            return np.mean(comp_k)
        for name_k in names:
            lines.append([data_i,clf_i,name_k,helper(name_k)])	
    cols=['dataset','clf','cs','quality']    
    df = pd.DataFrame(lines,columns=cols)
    df['alpha']=df['cs'].apply(get_alpha)
    df['cs']=df['cs'].apply(lambda x:x.split('_')[0])
    return df

in_path='pred'
clf_i='RF'
df=ens_quality(in_path,clf_i)
df_i=df[(df['cs']=='binary') &
        (df['alpha']=='0.25')]
print(df_i[['dataset','quality']].to_csv())

#df1=df[df['cs']=='multi']['quality']
#print(f'multi,{df1.mean():4f}')

#df1=df[(df['cs']=='weighted') &
#         (df['alpha']=='0.25') ]['quality']
#print(f'weighted/0.25,{df1.mean():.4f}')

#df1=df[(df['cs']=='weighted') &
#         (df['alpha']=='0.5') ]['quality']
#print(f'weighted/0.5,{df1.mean():.4f}')

#df1=df[(df['cs']=='binary') &
#         (df['alpha']=='0.25') ]['quality']
#print(f'binary/0.25,{df1.mean():.4f}')

#df1=df[(df['cs']=='binary') &
#         (df['alpha']=='0.5') ]['quality']
#print(f'binary/0.5,{df1.mean():.4f}')