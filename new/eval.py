import tools
tools.silence_warnings()
import argparse
import numpy as np
from time import time
from collections import defaultdict
import clfs,models,variants

class AllPreds(object):
    def __init__(self):
        self.pred={}#defaultdict(lambda :[])   
        self.true={}


def single_exp(data_path,model_path,result_path,p_value):
    X,y=tools.get_dataset(data_path)
    pred_dict= get_pred_dict(X,y,model_path)
    print(pred_dict.pred)

def get_pred_dict(X,y,model_path):
    clf_types=['RF','SVC']
    variant_types=['NECSCF','common']
    pred_dict=AllPreds() 
    model_reader=models.ManyClfs(model_path)
    for i,model_dict_i,train_i,test_i in model_reader.split(X,y):
        pred_dict.true[i]=test_i[1]
        pred_dict.pred[i]={}
        for name_j,pred_j in pred_iter(model_dict_i,
                                       train_i,test_i,
                                       clf_types,variant_types):
            pred_dict.pred[i][name_j]=pred_j
    return pred_dict

def pred_iter(model_dict_i,train_i,test_i,
        clf_types,variant_types):
    for name_j,nn_j in model_dict_i.items(): 
        if(clfs.is_cpu(nn_j)):
            ens_j=variants.make_ensemble(nn_j,train_i,test_i)
            for variant in  variant_types:
                for clf in clf_types:
                    pred_j=ens_j(clf,variant)
                    yield f'{variant}({clf})',pred_j
        else:
            pred_j=nn_j.predict(test_i[0])
            yield 'NECSCF(NN-TF)',pred_j

#def pred_iter(train_i,nn_j,clf_types):
#    for clf_type_j in clf_types:
#        clf_j=learn.get_clf(clf_type_j)
#        clf_j.fit(*train_i)
#        yield clf_type_j,clf_j    
#    if(clfs.is_cpu(nn_j)):
#        X=train_i[0]	
#        binary_j=nn_j.binary_model.predict(X)
         
#    else:
#        yield 'NECSCF(NN-TF)',nn_j

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='uci/vehicle')
    parser.add_argument("--models", type=str, default='vehicle/models')
    parser.add_argument("--results", type=str, default='vehicle/results2')
    parser.add_argument("--p_value", type=str, default='vehicle/p_value')
    parser.add_argument("--log_path", type=str, default='vehicle/log.time')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()
    tools.start_log(args.log_path)
    start=time()
    single_exp(args.data,args.models,args.results,args.p_value)
    tools.log_time(f'EVAL:{args.data}',start) 