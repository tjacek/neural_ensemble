from configparser import ConfigParser

def read_conf(in_path):
    config_obj = ConfigParser()
    config_obj.read(in_path)
    return config_obj['NCSCF']

def get_data_dir(clf_config):
    in_path=clf_config['in_path']
    if('datasets' in clf_config):
        datasets=clf_config['datasets'].split(',')
        return [ f'{in_path}/{path_i}' 
                  for path_i in datasets]
    else:
        return in_path

def save_result(clf_config,result_text):
    f = open(clf_config['out_path'],"w")
    f.write(result_text)
    f.close()