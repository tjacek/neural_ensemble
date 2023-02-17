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
            print(args)
            k= is_object(args)
            in_path= args[k]
            if(as_dict):
                output={}
            else:
                output=[]
            for path_i in data.top_files(in_path):
                print(path_i)
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
            k=is_object(args)
            if(dir_path is None):
                main_path=args[k]
            else:
                main_path=f'{args[k]}/{dir_path}'
            for path_i in data.top_files(main_path):
                args=list(args)
                args[k]=path_i
                result_i=fun(*args,**kwargs)
                results.append(result_i)
            full_results=learn.unify_results(results)
            acc= full_results.get_acc()
            if(show):
                print(acc)
            return acc  
        return decor_fun
    return helper

def is_object(args):
    if(type(args[0])==str):
        return 0
    return 1