import matplotlib.pyplot as plt
import pandas as pd 

def find_best(in_path):
    result_df=pd.read_csv(in_path)   
    best_result={}
    for data_i,row_i in best_gen(result_df):
        best_result[data_i]=(row_i['mean_acc'],row_i['std_acc'])
    return best_result

def best_gen(result_df):
    dataset=result_df['dataset'].unique()
    ens_df=result_df[result_df['ens']=='NECSCF']
    for data_i in dataset:
        df_i=ens_df[ens_df['dataset']==data_i]
        k=df_i['mean_acc'].argmax()
        yield data_i,df_i.iloc[k]

def box_plot(best_dict):
    fig, ax = plt.subplots()
    names=best_dict.keys()
    for i,value_i in enumerate(best_dict.values()):
        mean_i,std_i=value_i
        plt.errorbar(i, mean_i, yerr = std_i,
        	capsize=7,elinewidth=10,ecolor='b')
    plt.xticks(range(len(names)),names)
    plt.ylabel('Acc')
    plt.xlabel('Dataset')   
    plt.title('NECSCF')
    plt.show()

if __name__ == "__main__":
    in_path='../uci_bayes/bayes/result.csv'
    best=find_best(in_path)
    box_plot(best)