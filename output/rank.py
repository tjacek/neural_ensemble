import pandas as pd

def comput_rank(result_path):
    result_df=pd.read_csv(result_path)
    no_ensemble=set(['pca-only','common'])
    all_ranks=[]
    for data_i in result_df['dataset'].unique():
        df_i= result_df[result_df['dataset']==data_i]
        order_i= df_i['acc_mean'].argsort()
        rank_j=[]
        for j in order_i:
            row_j=df_i.iloc[[j]]
            var_j=row_j[['ens_type','clf_type','acc_mean']]
            var_j=var_j.to_numpy().tolist()
            var_j=[str(v) for v in var_j[0]]
            if(var_j[0] in no_ensemble):
                var_j[0]=f' {var_j[0]}'
            rank_j.append(','.join(var_j))
        rank_j.reverse()
        all_ranks.append('\n{}:\n{}'.format(data_i,'\n'.join(rank_j)))
    print('\n'.join(all_ranks))

if __name__ == "__main__":
    result_path= '../../uci/ova_hyper/result.csv'
    comput_rank(result_path)