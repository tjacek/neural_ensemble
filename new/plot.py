import tools
tools.silence_warnings()
import argparse
import pandas as pd
import numpy as np
from sklearn import manifold
from scipy import stats
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
import models,pred,variants

#def plot_binary(data_path,model_path,out_path):
#    binary_path=f'{model_path}/models/0'
#    tools.make_dir(out_path)
#    colors = ['red', 'green', 'blue', 'orange', 'purple']
#    for name_i,binary_i,y_test in binary_iter(data_path,binary_path):
#        path_i=f'{out_path}/{name_i}'
#        tools.make_dir(path_i)
#        for binary_j in binary_i:
#            tsne_j = manifold.TSNE(
#                n_components=2,
#                init="random",
#                random_state=0,
#                learning_rate="auto",
#                n_iter=300,
#            )
#            low_dim= tsne_j.fit_transform(binary_j)
#            for t, label in enumerate(y_test):
#                x,y=low_dim[t,0],low_dim[t,1]
#                plt.scatter(x,y,color=colors[label])
#                plt.annotate(label, (x, y))
#            plt.show()

def indv_acc(data_path,model_path,clf='RF'):
    X,y=tools.get_dataset(data_path)
    modelsIO=models.ManyClfs(model_path)
    for i,clf_dict_i,train_i,test_i in modelsIO.split(X,y): 
        for name_j,model_j in clf_dict_i.items():
            print(name_j)
            ens_j=variants.make_ensemble(model_j,train_i,test_i,None)
            clfs=variants.train_clfs(clf,ens_j.train)
            votes=variants.eval_clfs(clfs,ens_j.test)
            dict_j={}   
            for t,vote_t in enumerate(votes):
                pred_t=np.argmax(vote_t,axis=1)
                acc_t= accuracy_score(pred_t,ens_j.get_true())
                dict_j[t]=acc_t
            pred_j=variants.common_variant(ens_j,clf)
            dict_j['common']=accuracy_score(pred_j,ens_j.get_true())
            yield dict_j


def binary_iter(data_path,binary_path):
    clf_dict,(train_i,test_i)=models.single_read(binary_path)
    X,y=tools.get_dataset(data_path)
    X_train,y_train=X[train_i],y[train_i]
    X_test,y_test=X[test_i],y[test_i]
    for name_i,clf_i in clf_dict.items():
        binary_i= clf_i.binary_model.predict(X_test)
        yield name_i,X_test,binary_i,y_test

def save_indv(data_path,model_path,out_path,clf='RF'):
    acc=list(indv_acc(data_path,model_path,clf))
    with open(out_path, 'w') as f:
        json.dump(acc, f,cls=pred.NumpyEncoder)

def indiv_pvalue(in_path):
    with open(in_path, 'r') as f:
        indv_acc = json.load(f)
        keys=indv_acc[0].keys()
        samples_dict={key_i:[] for key_i in keys }
        for indv_dict_i in indv_acc: 
            for key_j in indv_dict_i:
                samples_dict[key_j].append(indv_dict_i[key_j])
        common_acc=samples_dict['common']
        del samples_dict['common']
        mean_acc=[ (key_i,np.mean(acc_i)) 
            for key_i,acc_i in samples_dict.items()]
        cat,acc=list(zip(*mean_acc))
        best=  cat[np.argmax(acc)]
        worst=  cat[np.argmin(acc)]
        p_best= get_pvalue(samples_dict[best],common_acc)
        p_worst= get_pvalue(samples_dict[worst],common_acc)
        mean_dict= dict(zip(cat,acc))
        return f'{np.mean(common_acc):.4f},{mean_dict[best]:.4f},{p_best},{mean_dict[worst]:.4f},{p_worst}'

def get_pvalue(x,y):
    r=stats.ttest_ind(x,y, equal_var=False)
    return round(r[1],4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../uci/vehicle')
    parser.add_argument("--models", type=str, default='../../mult_acc/vehicle/models')
#    parser.add_argument("--out", type=str, default='binary')
    args = parser.parse_args()
#    plot_binary(args.data,args.models,args.out)
#    save_indv(args.data,args.models,'indiv_acc.txt')
    indiv_pvalue('indiv_acc.txt')