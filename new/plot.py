import tools
tools.silence_warnings()
import argparse
import pandas as pd
from sklearn import manifold
import matplotlib.pyplot as plt
import models

def plot_binary(data_path,model_path,out_path):
    binary_path=f'{model_path}/models/0'
    tools.make_dir(out_path)
    for name_i,binary_i,y_test in binary_iter(data_path,binary_path):
        path_i=f'{out_path}/{name_i}'
        tools.make_dir(path_i)
        for binary_j in binary_i:
            tsne_j = manifold.TSNE(
                n_components=2,
                init="random",
                random_state=0,
                learning_rate="auto",
                n_iter=300,
            )
            low_dim= tsne_j.fit_transform(binary_j)
            print(low_dim[0,:].shape)
            plt.scatter(low_dim[:,0],low_dim[:,1])
            for t, label in enumerate(y_test):
                plt.annotate(label,(low_dim[t,0],low_dim[t,1]))
            plt.show()

def binary_iter(data_path,binary_path):
    clf_dict,(train_i,test_i)=models.single_read(binary_path)
    X,y=tools.get_dataset(data_path)
    X_train,y_train=X[train_i],y[train_i]
    X_test,y_test=X[test_i],y[test_i]
    for name_i,clf_i in clf_dict.items():
        binary_i= clf_i.binary_model.predict(X_test)
        yield name_i,binary_i,y_test
       	
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='imb/cleveland')
    parser.add_argument("--models", type=str, default='../../imb/cleveland')
    parser.add_argument("--out", type=str, default='binary')
    args = parser.parse_args()
    plot_binary(args.data,args.models,args.out)