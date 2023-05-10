import os,warnings
import numpy as np
import pandas as pd
import logging,time
from sklearn import preprocessing

def silence_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    tf.get_logger().setLevel('ERROR')
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

def start_log(log_path):
    logging.basicConfig(filename=log_path,#'{}/time.log'.format(dir_path), 
        level=logging.INFO,filemode='a', 
        format='%(process)d-%(levelname)s-%(message)s')

def log_time(txt,st):
    logging.info(f'{txt} took {(time.time()-st):.4f}s')

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

def get_dirs(path):
    return [path_i for path_i in top_files(path)
            if(os.path.isdir(path_i))]

def get_dataset(data_path):
    df=pd.read_csv(data_path,header=None) 
    return prepare_data(df)

def prepare_data(df):
    X=df.iloc[:,:-1]
    X=X.to_numpy()
    X=preprocessing.scale(X) # data preprocessing
    y=df.iloc[:,-1]
    cats={ cat_i:i for i,cat_i in enumerate(y.unique())}
    y=y.to_numpy()
    y=[cats[y_i] for y_i in y]
    return X,np.array(y)