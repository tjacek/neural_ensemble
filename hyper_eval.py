import pandas as pd
import dataset

in_path="neural/multi.csv"
df=pd.read_csv(in_path)
grouped=df.groupby(by='data')
metric="accuracy"
def helper(df_i):
    df_i=df_i.sort_values(by=metric,ascending=False)
    worst=df_i.iloc[-1]
    df_i["worst"]=worst[metric]
#    print(acc_i)
    return df_i.iloc[0]
df=grouped.apply(helper)
print(df)