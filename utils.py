from functools import wraps
import os.path
import data,learn

def iter_fun(n_iters=2):
    def decor_fun(fun):
        @wraps(fun)
        def iter_decorator(*args, **kwargs):
            out_path= args[1]
            data.make_dir(out_path)
            for i in range(n_iters):
                new_args=list(args)
                new_args[1]=f'{out_path}/{i}'
                fun(*new_args,**kwargs)
            return None
        return iter_decorator
    return decor_fun

def dir_fun(as_dict=False):    
    def helper(fun):
        @wraps(fun)
        def dir_decorator(*args, **kwargs):
#            print(args)
            k= is_object(args)
            in_path= args[k]
            if(as_dict):
                output={}
            else:
                output=[]
            paths=get_paths(in_path)
            for path_i in paths:
#                print(path_i)
                new_args=list(args)
                new_args[k]=path_i
                out_i=fun(*new_args,**kwargs)
                if(as_dict):
                    output[path_i.split('/')[-1]]=out_i
                else:
                    output.append(out_i)
            return output
        return dir_decorator
    return helper

def lazy_dir_fun(fun):
    @wraps(fun)
    def helper(*args, **kwargs):
        in_path,out_path=args[1],args[2]
        data.make_dir(out_path)
        for in_i in data.top_files(in_path):
            id_i=(in_i.split('/')[-1]).split('.')[0]
            out_i=f'{out_path}/{id_i}'
            if(not os.path.exists(out_i)):
                new_args=list(args)
                new_args[1]=in_i
                new_args[2]=out_i
                fun(*new_args,**kwargs)
        return None
    return helper

def unify_cv(dir_path='feats',show=False):
    def helper(fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            results=[]
            k,main_path=find_path(args,dir_path)
            for path_i in get_paths(main_path):
                args=list(args)
                args[k]=path_i
                result_i=fun(*args,**kwargs)
                results.append(result_i)
            if(type(results[0])==dict ):
                acc={}
                for key_i in results[0].keys():
                    by_key_i=[r_j[key_i] for r_j in results]
                    full_i=learn.unify_results(by_key_i)
                    acc[key_i]=full_i#.get_acc()
            else:
                full_results=learn.unify_results(results)
                acc= full_results#.get_acc()
            if(show):
                print(acc)
            return acc  
        return decor_fun
    return helper

def dir_map(depth=2, overwrite=False):
    def helper(fun):
        def rec_fun(in_path,out_path,counter=0):
            if(counter==depth):
                fun(in_path,out_path)
            else:
                data.make_dir(out_path)
                for in_i in data.top_files(in_path):
                    name_i=in_i.split('/')[-1]
                    out_i=f"{out_path}/{name_i}"                
                    if(overwrite or (not os.path.exists(out_i))):
                        rec_fun(in_i,out_i,counter+1)
                    else:
                        print(f'{out_i} exist')
        @wraps(fun)
        def decor_fun(in_path,out_path):
            rec_fun(in_path,out_path,0)
        return decor_fun
    return helper

def find_path(args,dir_path):
    k=is_object(args)
    if(dir_path is None):
        return k,args[k]
    else:
        return k,f'{args[k]}/{dir_path}'

def is_object(args):
    if(type(args[0])==str):
        return 0
    if(type(args[0])==list):
        return 0
    return 1

def get_paths(in_path):
    if(type(in_path)==list):
        return in_path
    else:
        return data.top_files(in_path)