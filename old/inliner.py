import tools
tools.silence_warnings()
import argparse
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import data,pred,variants

@tools.log_time(task='INLINER')
def single_exp(data_path,model_path,out_path):
    clfs=['RF','SVC']  
    X,y=data.get_dataset(data_path)
    pred_dict=defaultdict(lambda:[])
    inliner_voting=InlinerVariant(clfs,
                                  thres=3,
                                  n_neighbors=3)
    for name_i,exp_i in pred.get_exps(model_path):
        if(exp_i.is_ens()):
            train_i,test_i=exp_i.get_features(X,y)
            for clf_k,pred_k in inliner_voting(train_i,test_i):
                id_k=f'{name_i}-{clf_k}' #'{clf_k}-inliner'
                print(id_k)
                pred_dict[id_k].append((pred_k,test_i.y))
    tools.make_dir(out_path)
    for name_i,pred_i in pred_dict.items():
        pred.save_pred(f'{out_path}/{name_i}',pred_i)

class InlinerVariant(object):
    def __init__(self,clfs,thres=3,n_neighbors=3):
        self.clfs=clfs
        self.thres=thres
        self.n_neighbors=n_neighbors

    def __call__(self,train_i,test_i):
        neigh = KNeighborsClassifier(n_neighbors=3)
        y_near=[]
        for j,cs_j in enumerate(train_i.cs):
            neigh.fit(cs_j,train_i.y)
            y_near.append(neigh.predict(test_i.cs[j]))
        y_near=list(zip(*y_near))
        n_samples=test_i.y.shape[0]
        for clf_j in self.clfs:
            votes=variants.necscf(train_i,test_i,clf_j,True)
            votes=[[ vote_t[k,:] for vote_t in votes]
                    for k in range(n_samples)]
            y_pred=[]
            for k,vote_k in enumerate(votes):
                pred_k=np.argmax(vote_k,1)
                s_vote=[ vote_k[t] 
                     for t,(pred,near) in enumerate(zip(pred_k,y_near[k]))
                         if(pred==near)]
                if(len(s_vote)<self.thres):
                    s_vote=vote_k
                s_vote=np.sum(s_vote,axis=0)
                y_pred.append(np.argmax(s_vote,axis=0))
            yield f'{clf_j}-inliner',np.array(y_pred)

if __name__ == '__main__':
    dir_path='../optim_alpha/r_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../r_uci')
    parser.add_argument("--models", type=str, default=f'{dir_path}/models')
    parser.add_argument("--pred", type=str, default=f'{dir_path}/pred')
    parser.add_argument("--log", type=str, default='log.info')
    parser.add_argument("--dir", type=int, default=1)
    args = parser.parse_args()
    tools.start_log(args.log)
    if(args.dir>0):
        single_exp=tools.dir_fun(3)(single_exp)
    single_exp(args.data,args.models,args.pred)