import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def plot_dim( in_path,
              metric="accuracy",
              feats="info"):
    df=pd.read_csv(in_path)
    df=dataset.DFView(df)
    for df_i in df.by_data(sort=metric):
        feat_i=df_i[df_i["feats"]==feats]
        units=feat_i["dims"]*feat_i["layer"]
        units=units.tolist()
        metric_values=feat_i[metric].tolist()
        data_i=feat_i.iloc[0]["data"]
#        print(data_i)
        plt.scatter(units, metric_values)
        plt.title(data_i)
        plt.show()

in_path="neural/uci/raw_hyper.csv"
plot_dim(in_path,metric="accuracy")
#print(df)