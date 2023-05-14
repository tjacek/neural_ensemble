import tools
tools.silence_warnings()
import argparse
import numpy as np
from time import time
from collections import defaultdict
import json
import clfs,models,variants,learn

class AllPreds(object):
    def __init__(self):
        self.pred={}  
        self.true={}

    def compute_metric(self,metric_type='acc'):
        metric=learn.get_metric(metric_type)
        metrict_dict=defaultdict(lambda :[])
        for i,pred_dict in self.pred.items():
            true_i=self.true[i]
            for name_j,pred_j in pred_dict.items():
                metrict_dict[name_j].append(metric(true_i,pred_j) )
        return metrict_dict

    def save(self,out_path):
        raw_dict={'pred':self.pred,'true':self.true}
        with open(out_path, 'w') as f:
            json.dump(raw_dict, f,cls=NumpyEncoder)

    def select(self,indexes):
        if(type(indexes)==str):
            indexes=select_acc(indexes)
        all_pred=AllPreds()
        all_pred.pred={i:self.pred[str(i)] for i in indexes}
        all_pred.true={i:self.true[str(i)] for i in indexes}
        return all_pred

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read_acc_dict(acc_path):
    with open(acc_path, 'r') as f:
        acc_dict={}
        for line_i in f.readlines():
            raw=line_i.split(',')
            acc_dict[int(raw[0])]=eval(','.join(raw[1:]))
        return acc_dict

def select_acc(acc_dict):
    if(type(acc_dict)==str ):
        acc_dict=read_acc_dict(acc_dict)
    def helper(dict_i):
        values=list(dict_i.values())
        return (np.mean(values)>0.75)
    indexes=[i for i,dict_i in acc_dict.items()
                if(helper(dict_i))]
    return indexes

def read_preds(in_path):
    with open(in_path, 'r') as f:
        raw_dict = json.load(f)
        pred_dict=AllPreds()
        pred_dict.pred=raw_dict['pred']
        pred_dict.true=raw_dict['true']
        return pred_dict

def single_exp(data_path,model_path,out_path):
    X,y=tools.get_dataset(data_path)
    pred_dict= get_pred_dict(X,y,model_path)
    metric_dict=pred_dict.compute_metric()
    stats(metric_dict)
    pred_dict.save(result_path)

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

def stats(metric_dict):
    for name_i,metric_i in metric_dict.items():
        print(f'{name_i},{np.mean(metric_i):.4f},{np.std(metric_i):.4f}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='uci/vehicle')
    parser.add_argument("--models", type=str, default='vehicle/models')
    parser.add_argument("--out", type=str, default='vehicle/pred')
    parser.add_argument("--log_path", type=str, default='vehicle/log.time')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()
    tools.start_log(args.log_path)
    start=time()
    single_exp(args.data,args.models,args.out)
    tools.log_time(f'EVAL:{args.data}',start) 