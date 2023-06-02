import tools
tools.silence_warnings()
import argparse
import pandas as pd
import numpy as np
from sklearn import manifold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import models,variants

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
            for i,vote_i in enumerate(votes):
                pred_i=np.argmax(vote_i,axis=1)
                acc_i= accuracy_score(pred_i,ens_j.get_true())
                print(acc_i)
#    feats_i= list(binary_iter(data_path,model_path))[0]
#    name_i,common_i,binary_i,y_test= feats_i
#    np.concatenate([common_i,binary_i],axis=0)

def binary_iter(data_path,binary_path):
    clf_dict,(train_i,test_i)=models.single_read(binary_path)
    X,y=tools.get_dataset(data_path)
    X_train,y_train=X[train_i],y[train_i]
    X_test,y_test=X[test_i],y[test_i]
    for name_i,clf_i in clf_dict.items():
        binary_i= clf_i.binary_model.predict(X_test)
        yield name_i,X_test,binary_i,y_test
       	
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='uci/vehicle')
    parser.add_argument("--models", type=str, default='vehicle/models')
    parser.add_argument("--out", type=str, default='binary')
    args = parser.parse_args()
#    plot_binary(args.data,args.models,args.out)
    indv_acc(args.data,args.models)