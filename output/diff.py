import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))
import pandas as pd

def comp(default_path,paths):
    default=pd.read_csv(default_path)
    df={ id_i:pd.read_csv(path_i) 
         for id_i,path_i in paths.items()}
    datasets= default['dataset'].unique()
    clfs = default['clf_type'].unique()
    lines=[]
    for data_i in datasets:
        for clf_j in clfs:
            acc_ij=get_acc(data_i,clf_j,default)
            for optim_t,df_t in df.items():
                diff_t= get_acc(data_i,clf_j,df_t)-acc_ij 
#                line_t=f'{data_i},{clf_j},{optim_t},{diff_t:0.2}'
                line_t= data_i,clf_j,optim_t,diff_t
                lines.append(line_t)#.split(','))
    cols=['dataset','clf','optim','diff']
    df= pd.DataFrame(lines,columns=cols)
    print(df[df['diff']>0])
    print(df[df['diff']<0])

def get_acc(data_i,clf_j,df):
    row_ij=df[(df['dataset']==data_i) 
        & (df['clf_type']==clf_j)]
    row_ij=row_ij[row_ij['ens_type']=='NECSCF']
    return float(row_ij['mean_acc'])


in_path='../uci/default/result.csv'
paths={'bayes':'../uci/bayes/result.csv',
       'grid':'../uci/grid/result.csv'}

comp(in_path,paths)