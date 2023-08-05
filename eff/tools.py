import os,warnings
import logging,time
import tensorflow as tf
from sklearn.metrics import accuracy_score

def get_metric(metric_type):
    if(metric_type=='acc'):
	    return accuracy_score

def silence_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    paths=sorted(paths)
    return paths