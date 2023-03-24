import argparse,os,logging,warnings,time
from configparser import ConfigParser

def read_conf(in_path,dict_types,dir_path=None):
    if(type(dict_types)==str):
        dict_types=[dict_types]
    config_obj = ConfigParser()
    config_obj.read(in_path)
    parse_dict={'clf':parse_clf,
                'dir':parse_dir,
                'hyper':parse_hyper}
    full_dict={}
    for type_i in dict_types:
        dict_i=parse_dict[type_i](config_obj,
            dir_path)
        full_dict.update(dict_i)
    return full_dict 
        
def parse_clf(config_obj,dir_path=None):
    raw_dict= config_obj['CLF']
    clf_dict={ key_i:raw_dict[key_i].split(',') 
                for key_i in raw_dict}
    clf_dict['binary_type']=clf_dict['binary_type'][0]
    return clf_dict

def parse_dir(config_obj,dir_path=None):
    raw_dict= config_obj['DIR']
    if(dir_path is None):
        dir_path=raw_dict['main_dict']
    paths={'json':raw_dict['json'],
            'main_dict':dir_path}
    for key_i in raw_dict:
        if(key_i!='main_dict' and key_i!='json'):
            paths[key_i]=f'{dir_path}/{raw_dict[key_i]}'
    return paths

def parse_hyper(config_obj,dir_path=None):
    raw_dict= config_obj['HYPER']
    hyper= raw_dict['hyperparams'].split(',')
    hyper_dict={'hyperparams':hyper}
    for name_i in hyper:
        hyper_dict[name_i]=parse_list(raw_dict[name_i])
        default_i=f'default_{name_i}'
        hyper_dict[default_i]=raw_dict[default_i]
    hyper_dict['optim_type']=raw_dict['optim_type']
    hyper_dict['verbosity']=bool(raw_dict['verbosity'])
    hyper_dict['bayes_iter']=int(raw_dict['bayes_iter'])
    hyper_dict['n_jobs']=int(raw_dict['n_jobs'])
    return hyper_dict

def parse_list(raw_str):
    list_i=[]
    for item in raw_str.split(','):
        if(item.isdigit()):
            list_i.append(int(item))
        elif( item.replace(".","").isdigit()):
            list_i.append(float(item))
        else:
            list_i.append(item)
    return list_i

def parse_args(default_conf='conf/l1.cfg'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--conf",type=str,default=default_conf)
    parser.add_argument("--dir_path",type=str)
    parser.add_argument("--lazy",action='store_true')
    parser.add_argument("--default",action='store_true')
    parser.add_argument('--optim_type',
        choices=[ 'bayes', 'grid','conf'],default='conf')
    args = parser.parse_args()
    return args

def silence_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    tf.get_logger().setLevel('ERROR')
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

def log_time(txt,st):
    logging.info(f'{txt} took {(time.time()-st):.4f}s')