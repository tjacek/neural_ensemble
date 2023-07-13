import tools
tools.silence_warnings()
import argparse
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
import data,pred

@tools.log_time(task='INLINER')
def single_exp(data_path,model_path,out_path):
    clfs=['RF']#,'SVC']
    X,y=data.get_dataset(data_path)
    pred_dict=defaultdict(lambda:[])
    for name_i,exp_i in pred.get_exps(model_path):
        if(exp_i.is_ens()):
            train_i,test_i=exp_i.get_features(X,y)
            for clf_k,pred_k in inliner_voting(train_i,test_i,clfs):
                id_k=f'{name_i}-{clf_k}-inliner'
                pred_dict[id_k].append((pred_k,test_i.y))
    tools.make_dir(out_path)
    for name_i,pred_i in pred_dict.items():
        pred.save_pred(f'{out_path}/{name_i}',pred_i)

def inliner_voting(train_i,test_i,clfs):
    neigh = KNeighborsClassifier(n_neighbors=3)
    y_near=[]
    for cs_j in train_i.cs:
        neigh.fit(cs_j,train_i.y)
        y_near.append(neigh.predict(test_i.X))
    y_near=list(zip(*y_near))
    n_samples=test_i.y.shape[0]
    for clf_j in clfs:
        votes=pred.necscf(train_i,test_i,clf_j,True)
        votes=[[ vote_t[k,:] for vote_t in votes]
                    for k in range(n_samples)]
        y_pred=[]
        for k,vote_k in enumerate(votes):
            print(y_near[k])
            pred_k=np.argmax(vote_k,1)
            s_vote=[ vote_k[t] 
                     for t,(pred,near) in enumerate(zip(pred_k,y_near[k]))
                         if(pred==near)]
            if(len(s_vote)==0):
                s_vote=vote_k
            s_vote=np.sum(s_vote,axis=1)
            y_pred.append(np.argmax(s_vote,axis=0))
        yield clf_j,np.array(y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../data/cmc')
    parser.add_argument("--models", type=str, default='../test3/models/cmc')
    parser.add_argument("--pred", type=str, default='../test3/pred/cmc')
    parser.add_argument("--log", type=str, default='log.info')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    tools.start_log(args.log)
    if(args.dir>0):
        single_exp=tools.dir_fun(3)(single_exp)
    single_exp(args.data,args.models,args.pred)