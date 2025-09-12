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

in_path="neural/multi.csv"
df=best_param(in_path,metric="accuracy")
print(df)