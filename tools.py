import os,warnings
import logging#,time
from functools import wraps

def silence_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    tf.get_logger().setLevel('ERROR')
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

def dir_fun(fun):
    @wraps(fun)
    def helper(*args, **kwargs):
        in_path= args[0]
        for path_i in top_files(in_path):
            new_args=list(args)
            new_args[0]=path_i#f'{out_path}/{i}'
            fun(*new_args,**kwargs)
    return helper

def top_files(path):
    if(type(path)==str):
        paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    else:
        paths=path
    paths=sorted(paths)
    return paths

def round_data(data,decimals=4):
    if(type(data)==dict):
        return {name_i:round(value_i,4) 
                for name_i,value_i in data.items()}
    if(type(data)==list):
        return [round(value_i,4) for value_i in data]
    return round(data,4)

def get_dirs(path):
    return [path_i for path_i in top_files(path)
            if(os.path.isdir(path_i))]
@dir_fun
def test_fun(in_path):
    print(in_path)

if __name__ == "__main__":
    test_fun('data')