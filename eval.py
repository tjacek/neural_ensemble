import tools
tools.silence_warnings()
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from  summary.metric_dict import AccDictReader
import json

def single_exp(pred_path,out_path):

    acc_reader=AccDictReader(['inliner',
                              '-cs',
                              ['multi','base'],
                              'LR',
                              'RF'])
    acc_dict=acc_reader(pred_path,'acc') 
    df_i=acc_dict.to_df()
    df_i=df_i.sort_values(by='mean',
                            ascending=False)
    
    pvalue_df=summary(df_i,acc_dict)    
    df_i= df_i[['dataset','clf','cs','mean','std']]
    return df_i,pvalue_df

def summary(df_i,acc_dict):
    clf_dict=by_clf(df_i)
    id_dict={clf_i:to_ids( df_i)   
        for clf_i,df_i in clf_dict.items()}
    return sig_stats(id_dict,acc_dict)

def by_clf(df_i):
    return { clf_i:df_i[df_i['clf']==clf_i]
              for clf_i in  df_i['clf'].unique()}

def to_ids(df_i):
    base,other=None,[]
    for j,row in df_i.iterrows():
        id_i=','.join([row['dataset'],row['id']])
        if(row['cs']=='-'):
            base=id_i#row['id']
        else:
            other.append(id_i) #row['id'])
    return [base,other]

def sig_stats(id_dict,acc_dict):
    lines=[]
    for clf_i,ids_i in id_dict.items():
        base,other=ids_i
        for other_j in other:
            data_j,variant_j=other_j.split('-')[0].split(',')
            pvalue_j=acc_dict.pvalue(base,other_j)
            lines.append([data_j,clf_i,variant_j,pvalue_j])
    cols=['dataset','clf','class-specific','pvalue']
    pvalue_df=pd.DataFrame(lines,columns=cols)
    pvalue_df['sig']=pvalue_df['pvalue'].apply(lambda x:x<0.05)
    return pvalue_df

def to_latex(df_i):
    cols= df_i.columns.tolist()
    latext=' & '.join(cols)
    latext=f'\\hline {latext} \\\\ \n'
    for i,row_i in df_i.iterrows():
        row_i=[str(col_j) for col_j in list(row_i) ]
        line_i= ' & '.join(row_i)
        line_i=f'\\hline {line_i} \\\\ \n'
        latext+=line_i
    header= '|'.join([ 'l' for c in cols])
    header= '\\begin{tabular}{|'+header +'|}\n'
    latext=header + latext+'\\hline \n \\end{tabular}'
    latext=latext.replace('_','-')
    return latext

def sort_df(df_dict):
    subset={'sig_better':[],
            'insig_better':[],
            'sig_worse':[],
            'insig_worse':[]}
    for data_i,result_i in df_dict.items():
        sig_i=is_sig(result_i[1])
        better=improv(result_i[0])
        if(sig_i and better):
            subset['sig_better'].append(result_i)
        if((not sig_i) and better):
            subset['insig_better'].append(result_i)
        if(sig_i and (not better)):
            subset['sig_worse'].append(result_i)
        if((not sig_i) and (not better)):
            subset['insig_worse'].append(result_i)
    for subset_k in subset.values(): 
        df_i,pvalue_i=zip(*subset_k)
        df_i = pd.concat(df_i)
        pvalue_i = pd.concat(pvalue_i)
        print(to_latex(df_i))
        print(to_latex(pvalue_i))
    for key_k,subset_k in subset.items():
        print(f'{key_k},{len(subset_k)}')

def is_sig(pvalue_i):
    sig_i=pvalue_i['sig'].tolist()
    sig_i=np.array(sig_i,dtype=int)
    return sum(sig_i)>0 

def improv(df_i):
#    df_j=df_j.sort_values(by='mean',
#                          ascending=False)
    best_dict=df_i.iloc[0].to_dict()
    return best_dict['cs']!='-'
#    print(best_dict)

#def summary(acc_dict,transform=None):
#    data_dict=acc_dict.by_clf(transform=transform)
#    df_dict={}#defaultdict(lambda:{})
#    for data,clf_dict in data_dict.items():
#        lines=[]
#        for clf_j,df_j in clf_dict.items():
#            inliner_j= df_j[ df_j['variant']=='inliner']                              
#            other_j= df_j[ df_j['variant']!='inliner']
#            lines+=[best(inliner_j),best(other_j)]
#        cols=['dataset','id', 'mean','std','clf','cs','variant']
#        s_df=pd.DataFrame(lines,
#                          columns=cols)
#        s_df=s_df.sort_values(by='mean',
#                              ascending=False)
#        s_df=add_pvalues(data,s_df,acc_dict)
#        print(s_df)
#        s_df=show_impr(s_df)
#        df_dict[data]=s_df
#    print(df_dict)
#    return df_dict




#def add_pvalues(data,s_df,acc_dict,verbose=False):
#    base_cls= s_df[s_df['variant']=='-']
#    id_dict={  row_i['clf']:row_i['id'] 
#             for i,row_i in base_cls.iterrows()}
#    def helper(x):
#        if(x.variant=='-'):
#            return 1.0
#        if(not x.clf in id_dict):
#            return '-'
#        base_id=f'{data},{id_dict[x.clf]}'
#        x_id=f'{data},{x.id}'
#        return  acc_dict.pvalue(base_id,x_id)
#    s_df['pvalue']=s_df.apply( helper,axis=1)
#    s_df.drop('id', inplace=True, axis=1)
#    s_df=s_df[s_df['pvalue']!='-']
#    s_df['sig']=s_df['pvalue'].apply(lambda p: p<0.05)
#    if(verbose):
#        print(s_df.to_latex())
#    return s_df

#def show_impr(s_df):
#    base_cls= s_df[s_df['variant']=='-']
#    mean_dict={  row_i['clf']:row_i['mean'] 
#             for i,row_i in base_cls.iterrows()}
#    def helper(x):
#        if(x.variant=='-'):
#            return 0
#        return np.sign(x['mean']-mean_dict[x.clf] )
#    s_df['diff']=s_df.apply( helper,axis=1)
#    return s_df

if __name__ == '__main__':
    dir_path='../'
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default=f'{dir_path}/pred')
    args = parser.parse_args()
    if(os.path.isdir(args.pred)):
        single_exp=tools.dir_fun(2)(single_exp)
    df_dict=single_exp(args.pred,'out')
#    print(df_dict)
    sort_df(df_dict)