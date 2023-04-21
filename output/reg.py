import numpy as np
import pandas as pd 
from sklearn import linear_model
from scipy import stats
#import statsmodels.api as sma
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
import tools

def reg_exp(in_path,stats_path,base='common'):
    result_dict=tools.get_variant_results(in_path)
    result_dict.transform()
    result_dict.to_diff(base)
    rows=result_dict.as_rows()
    df=pd.DataFrame(rows[1:],columns=rows[0])
    full_df= reg_frame(df,stats_path)
    variants=result_dict.variant_names()
    reg_eval(full_df,variants)

#def reg_eval(full_df,variants):
#    features=['classes','samples','features','gini']
#    print( ','.join(['dataset','clf']+features))
#    for var_i in variants:
#        for clf_j in ['RF','NN-TF']:
#            df_j=full_df[full_df.clf==clf_j]
#            coff=linear_reg(df_j,features,var_i,robust=False)
#            line_ij=[var_i,clf_j]+[f'{c:.4f}' for c in coff]
#            print(','.join(line_ij))

def linear_reg(df,indep_var,dep_var ,robust=False):
    if(robust):
        clf=linear_model.HuberRegressor()
    else:
        clf=linear_model.LinearRegression()
    X=df[indep_var].to_numpy()
    y=df[dep_var].to_numpy()
    X=preprocessing.normalize(X,axis=0)
    clf.fit(X,y)
    return clf.coef_/np.sum(np.abs(clf.coef_))

def stats_frame(result,stats):
    if(type(result)==str):
        result=pd.read_csv(result)
    if(type(stats)==str):
        stats=pd.read_csv(stats)
    for col_i in stats.columns:
        def helper(data_j):
            index_j=(stats['dataset']==data_j)
            return stats[index_j][col_i].to_list()[0]
        result[col_i]=result['dataset'].apply(helper)
    result['samples']=result['samples'].apply(lambda x:np.log(x))
    return result

 
def pvalue_exp(in_path,base='common'):
    result_dict=tools.get_variant_results(in_path)
    variants=result_dict.variant_names()
    lines=[]
    cols=['variant','dataset','clf','diff','p_value','sig']
    for var_i in variants:
        if(var_i=='common'):
            continue
        for data_j,clf_j,dict_j in result_dict.iter():
            base_acc= dict_j[base]
            acc_j=dict_j[var_i]
            diff_i=np.mean(acc_j)-np.mean(base_acc)
            r=stats.ttest_ind(base_acc, acc_j, equal_var=False)
            p_value=r[1]
            line_i=[var_i,data_j,clf_j,diff_i,p_value,p_value<0.05]
            lines.append(line_i)
    df=pd.DataFrame(lines,columns=cols)
    return df

def pvalue_summary(df):
    var=df['variant'].unique()
    for var_i in var:
        df_i=df[df['variant']==var_i]
        better= df_i[(df['diff']>0) & (df['sig']== True)]
        neutral=df_i[ df['sig']== False]
        worse= df_i[(df['diff']<0) & (df['sig']== True)]
        print(f'{var_i},{len(better)},{len(worse)},{len(neutral)}')

def lda_exp(p_df,stats_path):
    full_df=stats_frame(p_df,stats_path)
#    def helper(row_i):
#       return int(row_i['clf']=='RF')
#    full_df['clf']=full_df.apply(func=helper,axis=1)
    def helper(row_i):
        diff=int(row_i['diff']>0)
        sign=int(row_i['sig'])
        return sign*(diff+1)
    full_df['y']=full_df.apply(func=helper,axis=1)
    features=['classes','samples','features','gini']
    y=full_df['y'].to_numpy()
    X=full_df[features].to_numpy()
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    print(features)
    print(clf.coef_)

if __name__ == "__main__":
    in_path='../uci/ova_hyper/output'
    stats_path='../uci/stats.csv'
    df=pvalue_exp(in_path)
    pvalue_summary(df)
    df_i= df[df['variant']=='NECSCF']
#    lda_exp(df_i,stats_path)