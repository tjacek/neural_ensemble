import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))
import numpy as np
from collections import defaultdict
import data,learn,utils

class VariantResults(dict):
    def __init__(self, arg=[]):
        super(VariantResults, self).__init__(arg)
    
    def variant_names(self):
        clf_i=list(self.values())[0]
        var_i=list(clf_i.values())[0]
        return var_i.keys()
    
    def iter(self):
        for data_i,dict_i in self.items():
            for clf_j,dict_j in dict_i.items():
                yield data_i,clf_j,dict_j

    def transform(self,fun=np.mean):
        for data_i,clf_j,dict_j in self.iter():
            for key_i,value_i in dict_j.items():
                dict_j[key_i]=fun(value_i)

    def to_diff(self,base='common'):
        for data_i,clf_j,dict_j in self.iter():
            base_value=dict_j[base]
            del dict_j[base]
            for key_i in dict_j:
                dict_j[key_i]-=base_value
                dict_j[key_i]/=base_value
    
    def as_rows(self):
        variants=self.variant_names()
        head=['dataset','clf']+list(variants)
        rows=[head]
        for data_i,clf_j,dict_j in self.iter():    
            line_i=[data_i,clf_j] 
            line_i+=[dict_j[var] for var in variants]
            rows.append(line_i)
        return rows

def read_results(output):
    @utils.dir_fun(as_dict=True)
    @utils.dir_fun(as_dict=True)
    def helper(path_i):
        return [ learn.read_result(path_j)
            for path_j in data.top_files(path_i)]
    return helper(output)

def get_variant_results(output):
    raw_dict=read_results(output)
    for data_i,dict_i in raw_dict.items():
        new_dict_i= defaultdict(lambda:{})
        for id_j,results_j in dict_i.items():
            variant_j,clf_j=id_j.split('_')
            new_dict_i[clf_j][variant_j]=get_acc(results_j)
        raw_dict[data_i]=new_dict_i
    return VariantResults(raw_dict)

def get_acc(results):
    return [result_j.get_acc() for result_j in results]