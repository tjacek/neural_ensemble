from configparser import ConfigParser

def read_conf(in_path):
    config_obj = ConfigParser()
    config_obj.read(in_path)
    dir_dict=  parse_dict(config_obj)
    hyper_dict=  parse_hyper(config_obj)
    return  {**dir_dict , **hyper_dict}

def parse_dict(config_obj):
    raw_dict= config_obj['DIR']
    dir_path=raw_dict['main_dict']
    paths={ key_i:f'{dir_path}/{key_i}' 
            for key_i in raw_dict
                if(key_i!='main_dict')}
    return paths

def parse_hyper(config_obj):
    hyper_dict={'hyper_optim':False}
    if('BAYES'in config_obj):
        raw_dict=config_obj['BAYES']
        hyper_dict['optim_type']='bayes'
        hyper_dict['hyper_optim']=True
        hyper_dict['n_jobs']=raw_dict['n_jobs']
        hyper_dict['verbosity']=bool(raw_dict['verbosity'])
    return hyper_dict

#def get_data_dir(clf_config):
#    in_path=clf_config['in_path']
#    if('datasets' in clf_config):
#        datasets=clf_config['datasets'].split(',')
#        return [ f'{in_path}/{path_i}' 
#                  for path_i in datasets]
#    else:
#        return in_path

def save_result(clf_config,result_text):
    f = open(clf_config['out_path'],"w")
    f.write(result_text)
    f.close()