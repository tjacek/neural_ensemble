import tools
tools.silence_warnings()
import numpy as np
from time import time
from collections import namedtuple

import pred,models,variants

def single_exp(data_path,model_path,out_path):
    clf_types=['RF','SVC']
    variant_types=['NECSCF','common']
    pred_dict=pred.AllPreds()
    for i,type_j,ens_j in ens_iter(data_path,model_path):
        pred_dict.true[i]=ens_j.get_true()
        pred_dict.pred[i]={}
        for variant in  variant_types:
            for clf in clf_types:
                pred_j=ens_j(clf,variant)
                name_i=f"{variant}({clf})"
                pred_dict.pred[i][name_i]=pred_j
    pred_dict.save(out_path)

def ens_iter(data_path,model_path):
    X,y=tools.get_dataset(data_path)
    dir_path= '/'.join(model_path.split('/')[:-1])
    acc_dict= pred.read_acc_dict(f'{dir_path}/acc.txt')
    modelsIO=models.ManyClfs(model_path)
    for i,clf_dict_i,train_i,test_i in modelsIO.split(X,y):             
        n_clf=len(acc_dict[i])
        for name_j,model_j in clf_dict_i.items():            
#            binary= model_j.binary_model.predict(test_i[0])
            threshold_i= 1.0 -(1/n_clf)
            s_clf=[k for k,acc_k in enumerate(acc_dict[i].values())
                        if(acc_k>threshold_i)]
            ens_j=variants.make_ensemble(model_j,train_i,test_i,s_clf)
            yield i,name_j,ens_j

if __name__ == '__main__':
    args=pred.parse_args()
    tools.start_log(args.log_path)
    start=time()
    single_exp(args.data,args.models,args.out)
    tools.log_time(f'PRED-SELECT:{args.data}',start) 
