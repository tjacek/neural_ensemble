import os,warnings
import logging,time
import argparse
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

class TimeLogging(object):
    def __init__(self,log_path):
        self.log = logging.getLogger(log_path)

    def __call__(self, fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            start_time = time.time()
            fun(*args, **kwargs)
            self.log.info( (time.time() - start_time))
        return decor_fun

class DirFun(object):
    def __init__(self,dir_args=None):
        if(dir_args is None):
            dir_args=[("in_path",0)]
        self.dir_args=dir_args

    def __call__(self, fun):#*args, **kwargs):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            in_path=self.get_input(*args, **kwargs)
            make_dir(self.get_output(*args, **kwargs))
            return_dict={}
            for in_i in top_files(in_path):
                id_i=in_i.split('/')[-1]
                new_args,new_kwargs=self.new_args(id_i,*args, **kwargs)
                return_dict[id_i]=fun(*new_args, **new_kwargs)
            return return_dict
        return decor_fun 
    
    def get_input(self,*args, **kwargs):
        name,i=self.dir_args[0]
        if(name in kwargs):
            return kwargs[name]
        return args[0]

    def get_output(self,*args, **kwargs):
        name,i=self.dir_args[-1]
        if(name in kwargs):
            return kwargs[name]
        return args[-1]

    def new_args(self,id_k,*args, **kwargs):
        new_args=list(args).copy()
        new_kwargs=kwargs.copy()
        for name_i,i in self.dir_args:
            if(name_i in kwargs):
                value_i=kwargs[name_i]
                new_kwargs[name_i]=f"{value_i}/{id_k}"
            else:
                print(i)
                value_i=args[i]
                new_args[i]=f"{value_i}/{id_k}"
        return tuple(new_args),new_kwargs

def print_dict(return_dict):
    for id_i,value_i in return_dict.items():
        print(f'{id_i},{value_i}')

def get_args(paths):
    parser = argparse.ArgumentParser()
#    parser.add_argument("--data", type=str)
#    parser.add_argument("--hyper", type=str)
    for path_i in paths:
        parser.add_argument(f"--{path_i}", type=str)
    parser.add_argument("--n_split", type=int, default=3)
    parser.add_argument("--n_iter", type=int, default=3)
    return parser