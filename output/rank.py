import pandas as pd
import numpy as np
from collections import defaultdict

def compute_rank(result_path,metric_col='mean'):
    result_df=pd.read_csv(result_path)
    ranks=defaultdict(lambda :[])
    for data_i in result_df['dataset'].unique():
        df_i= result_df[result_df['dataset']==data_i]
        df_i= df_i.sort_values(by=metric_col,ascending=False)
        for i,clf_i in enumerate(df_i['clf']):
            ranks[clf_i].append(i+1)
    print(borda_frame(ranks))

def borda_frame(ranks):
    lines=[]
    for data_i,rank_i in ranks.items():
        lines.append([data_i,borda_count(rank_i)])
    borda_df = pd.DataFrame(lines,columns=['dataset','borda'])
    borda_df= borda_df.sort_values(by='borda')
    return borda_df

def borda_count(ranks):
    return np.sum([(1/r) for r in ranks])

if __name__ == "__main__":
    result_path= 'result.csv'
    compute_rank(result_path)
