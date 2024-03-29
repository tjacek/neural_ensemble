import os,warnings
import logging,time
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

class DirFun(object):
    def __init__(self,dir_args=None):
        if(dir_args is None):
            dir_args=[("in_path",0)]
        self.dir_args=dir_args

    def __call__(self, fun):#*args, **kwargs):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            in_path=self.get_input(*args, **kwargs)
            for in_i in top_files(in_path):
                id_i=in_i.split('/')[-1]
                new_args,new_kwargs=self.new_args(id_i,*args, **kwargs)
                fun(*new_args, **new_kwargs)
        return decor_fun 
    
    def get_input(self,*args, **kwargs):
        name,i=self.dir_args[0]
        if(name in kwargs):
            return kwargs[name]
        return args[i]

    def new_args(self,id_k,*args, **kwargs):
        new_args=list(args).copy()
        new_kwargs=kwargs.copy()
        for name_i,i in self.dir_args:
            if(name_i in kwargs):
                value_i=kwargs[name_i]
                new_kwargs[name_i]=f"{value_i}/{id_k}"
            else:
                value_i=args[name_i]
                new_args[name_i]=f"{value_i}/{id_k}"
        return tuple(new_args),new_kwargs