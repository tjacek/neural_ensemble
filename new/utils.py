import os,warnings
from functools import wraps
import inspect
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
    
class DirFun(object):
    def __init__(self,main_path,path_args):
        self.main_path=main_path
        self.path_args=set(path_args)
    
    def __call__(self,fun):
        @wraps(fun)
        def decor_fun(*args,**kwargs):
            sig = inspect.signature(fun)
            keys= list(sig.parameters.keys())
            full_dict= dict(zip(keys,args))
            full_dict= full_dict | kwargs
            for arg_i in self.path_args:
                make_dir(full_dict[arg_i])
            paths=get_paths(full_dict[self.main_path])
            for path_i in paths:
                arg_i={self.main_path:path_i}
                id_i=path_i.split("/")[-1]
                for name_i,key_i in full_dict.items():
                    if(name_i==self.main_path):
                        continue
                    old_i=full_dict[name_i]
                    if(name_i in self.path_args):
                        arg_i[name_i]=f"{old_i}/{id_i}"
                    else:
                        arg_i[name_i]=old_i
                fun(**arg_i)
        return decor_fun

class DirProxy(object):
    def __init__(self,
                 main_path,
                 dirs):
        self.main_path=main_path
        self.dirs=dirs
        self.dict_paths={}

    def init(self):
        make_dir(self.main_path)
        for dir_i in self.dirs:
            path_i=f"{self.main_path}/{dir_i}"
            make_dir(path_i)
            self.dict_paths[dir_i]=path_i

    def __getitem__(self,name):
        return self.dict_paths[name]

    def count(self,name):
        path=self.dict_paths[name]
        return len(top_files(path))