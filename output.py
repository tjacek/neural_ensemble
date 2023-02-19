import numpy as np
import ens,utils

class ESCFExp(object):
    def __init__(self,ens_types=[ens.Ensemble],
        clf_types=['LR','RF'],ensemble_reader=None):
        if(ensemble_reader is None):
            ensemble_reader=ens.npz_reader
        self.ensemble_reader=ensemble_reader
        self.ens_types=ens_types
        self.clf_types=clf_types

    @utils.dir_fun(as_dict=True)
    def __call__(self,in_path):
        @utils.dir_fun(as_dict=False)
        @utils.unify_cv(dir_path=None)
        def helper(path_i):
            common,binary=self.ensemble_reader(path_i)
            result_dict={}
            for alg_i in self.ens_types:
                for clf_j in self.clf_types:
                    ens_i=alg_i(common,binary,clf_j)
                    result_i=ens_i.evaluate()
                    id_ij=",".join([str(ens_i),clf_j])
                    print(id_ij)
                    result_dict[id_ij]=result_i
            return result_dict
        acc=helper(in_path)    
        exp={ id_i:[acc_j[id_i] for acc_j in acc] 
                for id_i in acc[0].keys()}
        lines=[ f'{id_i},{stats(acc_i,True)}' 
            for id_i,acc_i in exp.items()]
        return lines

def stats(acc,as_str=False):
    raw=[f'{fun_i(acc):.4f}' 
        for fun_i in [np.mean,np.std]]
    if(as_str):
        return ','.join(raw)
    return raw

def format(line_dict):
    lines=[]
    for data_i,lines_i in line_dict.items():
        for line_j in lines_i:
            lines.append(f'{data_i},{line_j}')
    print('\n'.join(lines))

exp=ESCFExp()
line_dict=exp('test')
format(line_dict)