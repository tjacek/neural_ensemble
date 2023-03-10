from configparser import ConfigParser

def read_conf(in_path):
    config_obj = ConfigParser()
    config_obj.read(in_path)
    dir_dict=  parse_dict(config_obj)
    hyper_dict=  parse_hyper(config_obj)
    return  {**dir_dict , **hyper_dict}

def read_test(in_path):
    config_obj = ConfigParser()
    config_obj.read(in_path)
    dir_dict=parse_dict(config_obj)    
    clf_dict=  parse_clf(config_obj)
    return  {**dir_dict , **clf_dict}

def parse_clf(config_obj):
    raw_dict= config_obj['CLF']
    return { key_i:raw_dict[key_i].split(',') 
            for key_i in raw_dict}

def parse_dict(config_obj):
    raw_dict= config_obj['DIR']
    dir_path=raw_dict['main_dict']
    paths={ key_i:f'{dir_path}/{raw_dict[key_i]}' 
            for key_i in raw_dict
                if(key_i!='main_dict')}
    return paths

def parse_hyper(config_obj):
    hyper_dict={'hyper_optim':False}
    if('GRID'in config_obj):
        raw_dict=config_obj['GRID']
        hyper_dict['optim_type']='grid'
        hyper_dict['hyper_optim']=True
        hyper_dict['n_jobs']=int(raw_dict['n_jobs'])
    if('BAYES'in config_obj):
        raw_dict=config_obj['BAYES']
        hyper_dict['optim_type']='bayes'
        hyper_dict['hyper_optim']=True
        hyper_dict['n_jobs']=int(raw_dict['n_jobs'])
        hyper_dict['verbosity']=bool(raw_dict['verbosity'])
    return hyper_dict

def save_result(clf_config,result_text):
    f = open(clf_config['out_path'],"w")
    f.write(result_text)
    f.close()