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

def dir_fun(fun):
    @wraps(fun)
    def decor_fun(*args, **kwargs):
        in_path,out_path=kwargs['in_path'],kwargs['out_path']
        make_dir(out_path)
        for in_i in top_files(in_path):
            name_i=in_path.split('/')[-1]
            out_i=f'{out_path}/{name_i}'
            args_i=kwargs.copy()
            args_i['in_path']=in_i
            args_i['out_path']=out_i
            fun(*args ,**args_i)
        return fun(*args,**kwargs)
    return decor_fun