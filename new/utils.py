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
        paths=[]
        for path_i in path:
            paths+=top_files(path_i)
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

class PathSelect(object):
    def __init__( self,
                  perm_paths=None,
                  taboo_paths=None):
        if(type(perm_paths)==list):
            perm_paths=set(perm_paths)
        if(type(perm_paths)==list):
            perm_paths=set(perm_paths)
        self.perm_paths=perm_paths
        self.taboo_paths=taboo_paths

    def __call__(self,paths):
        if(self.perm_paths):
            return self.get_perm(paths)
        if(self.taboo_paths):
            return self.remove_taboo(paths)

    def get_perm(self,paths):
        new_paths=[]
        for path_i in paths:
            id_i=path_i.split("/")[-1]
            if(id_i in self.perm_paths):
                new_paths.append(path_i)
        return new_paths

    def remove_taboo(self,paths):
        new_paths=[]
        for path_i in paths:
            id_i=path_i.split("/")[-1]
            if(not id_i in self.taboo_paths):
                new_paths.append(path_i)
        return new_paths


class DirFun(object):
    def __init__(self,main_path,path_args=[]):
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
            output_args={}
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
                output_args[id_i]=fun(**arg_i)
            output_args={key:value
                          for key,value in output_args.items()
                              if(not (value is None))}    
            return output_args
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

def split_list(data,split_size):
    import math
    n_splits=math.ceil(len(data)/split_size)
    return [ data[i*split_size:(i+1)*split_size] 
                for i in range(n_splits)]

def as_latex( lines,
              cols):
    n_cols=len(cols)
    text="\\begin{table}[ht]\n"
    text+="\centering\n"
    header="|".join([ "c" for _ in range(n_cols)])
    text+="\\begin{tabular}{|"
    text+=header 
    text+="|}\n"
    text+= latex_line(cols)
    def helper(e):
        if(type(e)==str):
            return e
        else:
            return f"{e:.2f}"
    for line_i in lines:
        raw_i= [ helper(e_j) 
                    for e_j in line_i]
        text+= latex_line(raw_i)
    text+="\hline\n"
    text+="\end{tabular}\n"
    text+="\end{table}\n"
    return text

def latex_line(raw_i):
    raw_i=" & ".join(raw_i)
    return "\hline " + raw_i + "\\\\ \n" 