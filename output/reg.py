import numpy as np
import pandas as pd 
from sklearn import linear_model
from scipy import stats
import statsmodels.api as sma
from sklearn import preprocessing
import tools

def reg_exp(in_path,stats_path,base='common'):
    result_dict=tools.get_variant_results(in_path)
    result_dict.to_diff(base)
    rows=result_dict.as_rows()
    df=pd.DataFrame(rows[1:],columns=rows[0])
    full_df= reg_frame(df,stats_path)
    variants=result_dict.variant_names()
#    print(full_df['clf']=='RF')
    reg_eval(full_df,variants)

def reg_eval(full_df,variants):
#    df=reg_frame(result,stats)
    features=['classes','samples','features','gini']
    print( ','.join(['dataset','clf']+features))
    for var_i in variants:
        for clf_j in ['RF','NN-TF']:
            df_j=full_df[full_df.clf==clf_j]
            coff=linear_reg(df_j,features,var_i,robust=False)
            line_ij=[var_i,clf_j]+[f'{c:.4f}' for c in coff]
            print(','.join(line_ij))
#    stat_sig= p_value(df)
#    print(','.join(features))
#    print(coff)
#    print(stat_sig)

def reg_frame(result,stats):
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
    
def p_value(df):
    X=df[['classes','samples','features','gini']].to_numpy()
    y=df['diff'].to_numpy()
    X=preprocessing.normalize(X,axis=0)
    est  = sma.OLS(y, X)
    output  = est.fit()
#    print(f'pvalue:{output.f_pvalue}')
    return output.summary2().tables[1]['P>|t|']


if __name__ == "__main__":
    in_path='../uci/ova_hyper/output'
    stats_path='../uci/stats.csv'
    reg_exp(in_path,stats_path)
#    reg_exp(result_dict)
#result_path='diff.csv'
#stats_path='stats.csv'
#reg_eval(result_path,stats_path)