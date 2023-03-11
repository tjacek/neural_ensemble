import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def find_best(in_path):
    result_df=pd.read_csv(in_path)   
    best_result={}
    for data_i,row_i in best_gen(result_df):
        best_result[data_i]=(row_i['mean_acc'],row_i['std_acc'])
    return best_result

def best_gen(result_df,ens_type='NECSCF'):
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

def scatter_plot(df,x_col,y_col='diff'):
    x=df[x_col].to_numpy()
    y=df[y_col].to_numpy()
    labels=df.dataset.tolist()

    plt.figure()
    ax = plt.subplot(111)
    ax.set_ylim(get_limit(y)) 
    ax.set_xlim(get_limit(x))
    for i,name_i in enumerate(labels):    
#        color_i= np.random.rand(3)
        plt.text(x[i], y[i], name_i,#color=color_i,
                   fontdict={'weight': 'bold', 'size': 9})
#    ax.set_xticks(ticks)
    plt.grid()
    plt.ylabel(y_col)
    plt.xlabel(x_col)   
#    plt.title(title_name)
    plt.show()

def get_limit(series):
    delta=np.std(series)
    s_max=np.amax(series)
    s_min=np.amin(series)
    if(s_min>0):
        s_min=0
    else:
        s_min-= 0.1*delta 
    return [s_min,s_max+0.3*delta]


if __name__ == "__main__":
    in_path='../uci_bayes/bayes/result.csv'
    best=find_best(in_path)
    box_plot(best)