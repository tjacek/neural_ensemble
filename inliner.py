import tools
tools.silence_warnings()
import argparse
from sklearn.neighbors import KNeighborsClassifier
import data,pred

@tools.log_time(task='INLINER')
def single_exp(data_path,model_path,out_path):
    neigh = KNeighborsClassifier(n_neighbors=3)
    clfs=['RF','SVC']
    X,y=data.get_dataset(data_path)
    for name_i,model_i,split_i in pred.get_model_paths(model_path):
        if('ens' in name_i):
            train,test=split_i.get_dataset(X,y)
            cs_train=model_i.extract(train.X)
            cs_test=model_i.extract(test.X)
            for cs_j in cs_train:
                neigh.fit(cs_j,train.y)
                y_j = neigh.predict(test.X)
#            for clf_k in clfs:
#                pred_k=pred=necscf(train,test,cs_train,cs_test,clf_k)         print(y_j)

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