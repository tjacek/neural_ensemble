import tools
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
import data,learn,pred

def beter_acc(data_path,model_path,clf_type='RF'):
    X,y=data.get_dataset(data_path)
    better=[]
    for nn_i,split_i in read_models(model_path):
        train,test= split_i.get_dataset(X,y)
        common_acc= learn.fit_clf(train,test,
            clf_type,hard=True,acc=True)
        ens_acc= necscf(train,test,nn_i,clf_type)
        print((common_acc,ens_acc))
        if(ens_acc>common_acc):
            better.append(ens_acc)
    print(len(better))
#        full_train,full_test=get_full(train,nn_i),get_full(test,nn_i)
#        clf_i=learn.get_clf(clf_type)
#        clf_i.fit(train.X,train.y)
#        clf_i

def get_full(train ,nn):
    cs_train=nn.extract(train.X)
    return [np.concatenate([train.X,cs_i],axis=1)
             for cs_i in cs_train]

def read_models(in_path):
    for path_i in tools.top_files(in_path):
        yield pred.read_model(path_i)

def necscf(train,test,nn,clf_type):
    full_train,full_test=get_full(train,nn),get_full(test,nn)
    votes=[]
    for train_i,test_i in zip(full_train,full_test):
        clf_i=learn.get_clf(clf_type)
        clf_i.fit(train_i,train.y)
        y_pred=clf_i.predict_proba(test_i)
        votes.append(y_pred)
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    y_pred= np.argmax(votes,axis=1)
    return accuracy_score(y_pred,test.y)


if __name__ == '__main__':
    data_path="10_10/models/solar-flare/binary_ens(0.5)"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='data/solar-flare')
    parser.add_argument("--model", type=str, default=data_path)
    args = parser.parse_args()
    beter_acc(args.data,args.model)
