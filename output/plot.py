import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))
import conf
conf.silence_warnings()
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import conf,data,learn,utils,tools


def box_gen(conf):
    result_dict=tools.read_results(conf['output'])
    dataset=list(result_dict.keys())
    ens_types=result_dict[dataset[0]].keys()
    data.make_dir(conf['box'])
    for ens_i in ens_types:
        dict_i={ data_j:[ result_k.get_acc()  
            for result_k in result_dict[data_j][ens_i]]
                for data_j in dataset}
        box_plot(ens_i,dict_i,'{}/{}'.format(conf['box'],ens_i))

def box_plot(ens_i,dict_i,out_i=None,
        xlabel='Acc',ylabel='Dataset'):
    plt.clf()
    acc,labels = [],[]
    for label_i,acc_i in dict_i.items():
        acc.append(acc_i)
        labels.append(label_i)
    plt.boxplot(acc, labels=labels)
    plt.title(ens_i)
    plt.xlabel(xlabel)
    plt.xticks(rotation = 90)
    plt.ylabel(ylabel)
    if(not (out_i is None)):
        plt.savefig(f'{out_i}.png')
    return plt
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",type=str,default='conf/base.cfg')
    parser.add_argument("--dir_path",type=str)
    args = parser.parse_args()
    conf_dict = conf.read_conf(args.conf,'dir',args.dir_path)
    box_gen(conf_dict)