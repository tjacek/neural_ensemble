import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))
import numpy as np
from collections import defaultdict
import data,learn,utils

#class ResultDict(object):
#    def __init__(self,dataset,variants,clfs):
#    def __init__(self, arg=[]):
#        super(ResultDict, self).__init__(arg)

def read_results(output):
    @utils.dir_fun(as_dict=True)
    @utils.dir_fun(as_dict=True)
    def helper(path_i):
        return [ learn.read_result(path_j)
            for path_j in data.top_files(path_i)]
    return helper(output)

def make_result(output):
    raw_dict=read_results(output)
    for data_i,dict_i in raw_dict.items():
        new_dict_i= defaultdict(lambda:{})
        for id_j,results_j in dict_i.items():
            variant_j,clf_j=id_j.split('_')
            new_dict_i[clf_j][variant_j]=mean_acc(results_j)
        raw_dict[data_i]=new_dict_i
    print(raw_dict)

def mean_acc(results):
    return np.mean([result_j.get_acc() for result_j in results])