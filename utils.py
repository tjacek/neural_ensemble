import os,warnings
import logging,time
import argparse,json
from functools import wraps

def silence_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    tf.get_logger().setLevel('WARNING')#'ERROR')
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    if(type(path)==str):
        paths=[ f'{path}/{file_i}' for file_i in os.listdir(path)]
    else:
        paths=path
    paths=sorted(paths)
    return paths

def read_json(in_path):
    with open(in_path, 'r') as file:
        data = json.load(file)
        return data

def save_json(value,out_path):
    with open(out_path, 'w') as f:
        json.dump(value, f)

class DirFun(object):
    def __init__(self,
                 input_arg='in_path',
                 out_arg='out_path',
                 dir_args=None):
        if(dir_args is None):
            dir_args={input_arg:0}
        if(out_arg and (not out_arg in dir_args)):
            dir_args[out_arg]=len(dir_args)
        self.dir_args=dir_args
        self.input_arg=input_arg
        self.out_arg=out_arg

    def __call__(self, fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            original_args=FunArgs(args,kwargs)
            input_pair=self.get_pair(self.input_arg)
            in_path=original_args.get(input_pair)
            if(not os.path.isdir(in_path)):
                return fun(*args, **kwargs)
            if(self.out_arg):
                out_pair=self.get_pair(self.out_arg)
                out_path=original_args.get(out_pair)
                make_dir(out_path)
            result_dict={}
            for in_i in top_files(in_path):
                id_i=in_i.split('/')[-1]
                arg_i=original_args.copy()
                for pair_j in self.dir_args.items():
                    path_j=arg_i.get(pair_j)
                    arg_i.set(pair_j,f"{path_j}/{id_i}")
                result_dict[id_i]=fun(*arg_i.args,**arg_i.kwargs)
            return result_dict
        return decor_fun
    
    def get_pair(self,name):
        return name,self.dir_args[name]

class FunArgs(object):
    def __init__(self,args,kwargs):
        self.args=list(args)
        self.kwargs=kwargs
    
    def copy(self):
        return FunArgs(args=self.args.copy(),
                       kwargs=self.kwargs.copy())
    
    def get(self,pair):
        name,index=pair
        if(name in self.kwargs):
            return self.kwargs[name]
        else:
            return self.args[index]

    def set(self,pair,value):
        name,index=pair
        if(name in self.kwargs):
            self.kwargs[name]=value
        else:
            self.args[index]=value