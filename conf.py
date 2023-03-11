from configparser import ConfigParser

def read_hyper(in_path):
    config_obj = ConfigParser()
    config_obj.read(in_path)
    dir_dict=  parse_dict(config_obj)
    hyper_dict= parse_hyper(config_obj)
    return dir_dict,hyper_dict

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
    raw_dict= config_obj['HYPER']
    hyper= raw_dict['hyperparams'].split(',')
    hyper_dict={'hyperparams':hyper}
    for name_i in hyper:
        hyper_dict[name_i]=parse_list(raw_dict[name_i]) #.split(',')
    hyper_dict['optim_type']=raw_dict['optim_type']
    hyper_dict['verbosity']=bool(raw_dict['verbosity'])
    hyper_dict['bayes_iter']=int(raw_dict['bayes_iter'])
    hyper_dict['n_jobs']=int(raw_dict['n_jobs'])
    return hyper_dict

def parse_list(raw_str):
    list_i=[]
    for item in raw_str.split(','):
        if(item.isnumeric()):
            list_i.append(int(item))
        else:
            list_i.append(item)
    return list_i

#def save_result(clf_config,result_text):
#    f = open(clf_config['out_path'],"w")
#    f.write(result_text)
#    f.close()