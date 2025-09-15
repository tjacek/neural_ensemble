import numpy as np
import pandas as pd
import dataset

def best_param(in_path,metric="accuracy"):
    df=pd.read_csv(in_path)
    grouped=df.groupby(by='data')
    def helper(df_i):
        df_i=df_i.sort_values(by=metric,ascending=False)
        worst=df_i.iloc[-1]
        df_i["worst"]=worst[metric]
        return df_i.iloc[0]
    return grouped.apply(helper)

def var_param(in_path,metric="accuracy"):
    df=pd.read_csv(in_path)
    grouped=df.groupby(by='data')
    def helper(df_i):
        metric_i=df_i[metric]
        mean_i=np.mean(metric_i)
        std_i=np.std(metric_i)
        data_i=df_i['data'].to_list()[0]
        return pd.Series([data_i,mean_i,std_i])
    return grouped.apply(helper)

in_path="neural/uci/raw_hyper.csv"
df=var_param(in_path,metric="accuracy")
print(df)