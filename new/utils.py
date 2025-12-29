import os,warnings
import re 

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    if(type(path)==str):
        paths=[ f'{path}/{file_i}' for file_i in os.listdir(path)]
    else:
        paths=path
    paths=sorted(paths,key=natural_keys)
    return paths

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def get_paths(dir_paths,taboo=None):
    all_paths=[]
    if(type(dir_paths)==list):
        for dir_i in dir_paths:
            all_paths+=top_files(dir_i)
    else:
        all_paths+=top_files(dir_paths)
    if(taboo is None):
        return all_paths
    taboo=set(taboo)
    s_paths=[]
    for path_i in all_paths:
        id_i=path_i.split("/")[-1]
        if(not id_i in taboo):
            s_paths.append(path_i)
    return s_paths
    