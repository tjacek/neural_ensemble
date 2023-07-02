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

def dir_fun(n_paths=1):
    def helper(fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            path_args=args[:n_paths]
            make_dir(path_args[-1])
            for path_i in top_files(path_args[0]):
                name_i=path_i.split('/')[-1]
                new_args=list(args)
                for j,arg_j in enumerate(path_args):
                    new_args[j]=f'{arg_j}/{name_i}'
                fun(*new_args,**kwargs)
        return decor_fun
    return helper

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

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

def start_log(log_path):
    logging.basicConfig(filename=log_path,#'{}/time.log'.format(dir_path), 
        level=logging.INFO,filemode='a', 
        format='%(process)d-%(levelname)s-%(message)s')

def log_time(txt,st):
    logging.info(f'{txt} took {(time.time()-st):.4f}s')

@dir_fun
def test_fun(in_path):
    print(in_path)

if __name__ == "__main__":
    test_fun('data')