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

def plot_xy( in_path,
             series="layer",
             x_series=1,
             y_series=2,
             value="accuracy"):
    df=pd.read_csv(in_path)
    df=dataset.DFView(df)
    for df_i in df.by_data("accuracy"):
        data_i=df_i.iloc[0]["data"]
        x_i=df_i[df_i[series]==x_series][value].tolist()
        y_i=df_i[df_i[series]==y_series][value].tolist()
        plt.scatter(x_i, y_i)
        plt.title(data_i)
        plt.show()

def show_feat(in_path,desc_path):
    desc_df=dataset.csv_desc(desc_path)
    dim_dict=desc_df.get_dict("id","dims")
    df=pd.read_csv(in_path)
    df=dataset.DFView(df)
    for df_i in df.by_data("accuracy"):
        data_i=df_i["data"].unique()[0]
        dim_i=dim_dict[data_i]
        df_i["units"]=df_i.apply(lambda row:row.layer*(dim_i+row.dims),axis=1)
        df_i=df_i.sort_values(by="units")
        print(df_i)
    print(dim_dict)

if __name__ == '__main__':
    in_path="neural/uci/raw_hyper.csv"
    show_feat(in_path,"neural/uci/data")
#    df=plot_xy(in_path)#,metric="accuracy")
#    print(df)