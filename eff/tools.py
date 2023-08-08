import os,warnings
import logging,time
import re
from sklearn.metrics import accuracy_score
from functools import wraps

def get_metric(metric_type):
    if(metric_type=='acc'):
	    return accuracy_score

def silence_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    tf.get_logger().setLevel('ERROR')
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

def get_hyper(name_i,hyper_df):
    hyper_i=hyper_df[hyper_df['dataset']==name_i]
    hyper_i= hyper_i.iloc[0].to_dict()
    layers= [key_i for key_i in hyper_i
                   if('unit' in key_i)]
    layers.sort()
    hyper_i['layers']=[hyper_i[name_j] 
                          for name_j in layers]
    return hyper_i

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    if(type(path)==str):
        paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    else:
        paths=path
    paths.sort(key=natural_keys)# =sorted(paths)
    return paths

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def dir_fun(n_paths=1):
    def helper(fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            path_args=args[:n_paths]
            make_dir(path_args[-1])
            path_dir={}
            for path_i in top_files(path_args[0]):
                name_i=path_i.split('/')[-1]
                new_args=list(args)
                for j,arg_j in enumerate(path_args):
                    new_args[j]=f'{arg_j}/{name_i}'
                path_dir[path_i]= fun(*new_args,**kwargs)
            return path_dir
        return decor_fun
    return helper

def log_time(task='TRAIN'):
    def helper(fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            name_i=args[0].split('/')[-1]
            start=time.time()
            result=fun(*args,**kwargs)
            diff=(time.time()-start)
            logging.info(f'{task}-{name_i}-{diff:.4f}s')
            return result
        return decor_fun
    return helper

def start_log(log_path):
    logging.basicConfig(filename=log_path,#'{}/time.log'.format(dir_path), 
        level=logging.INFO,filemode='a', 
        format='%(process)d-%(levelname)s-%(message)s')