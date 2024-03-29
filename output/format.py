from collections import defaultdict
import re
import pandas as pd

result_path="result.csv"
result_df=pd.read_csv(result_path)
clf= result_df['clf'].unique()
types=['RF','SVC']
def helper(name_i):
    for type_j in types:
        if(type_j in name_i):
            return type_j
    return "NN"
result_df['type']=result_df['clf'].apply(helper)
types+=['NN']

df_dict={}
for type_j in types:
    df_i=result_df[result_df['type']==type_j]
    df_i['mean']=df_i['mean'].apply(lambda x:round(x ,4))
    df_i['std']=df_i['std'].apply(lambda x:round(x ,4))
    df_i.sort_values(by=['dataset','mean'])
    df_i.drop('type', inplace=True, axis=1)
    df_dict[type_j]=df_i

#nn_df=df_dict['NN']

#df_dict['NN']=nn_df[ (nn_df['clf']!='binary_ens-0.25') 
#                & (nn_df['clf']!='binary_ens-0.5')]

#print(df_dict['NN'].to_latex())

clf='SVC'

def get_alpha(raw):
    digits=re.findall(r'\d+',raw)
    if(len(digits)>0):
        return f'0.{digits[1]}'
    return '-'

def get_cs(raw):
    cs= raw.split('_')[0]
    if(cs==clf):
        return '-'
    return cs

df=df_dict[clf]
df['alpha']=df['clf'].apply(get_alpha)
df['cs']=df['clf'].apply(get_cs)
df['clf']=df['clf'].apply(lambda x:clf)
df=df[['dataset','cs','alpha','clf','mean','std']]
print(df.to_latex())

