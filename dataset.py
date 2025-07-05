import numpy as np
import pandas as pd
from sklearn import preprocessing
import utils

class Dataset(object):
    def __init__(self,X,y=None):
        self.X=X
        self.y=y
    
    def weight_dict(self):
        return get_class_weights(self.y)

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[0]
        
    def n_cats(self):
        return int(max(self.y))+1

    def gini(self):
        return self.weight_dict().gini()

    def proportion(self):
        weight_dict=self.weight_dict()
        values=list(weight_dict.values())
        return max(values)/min(values)

    def save_csv(self,out_path):
        with open(out_path,'w') as f:
            for i,y_i in enumerate(self.y):
                x_i=self.X[:,i]
                line_i=x_i.tolist()
                line_i.append(y_i)
                line_i=",".join([str(c_j) for c_j in line_i])
                line_i+="\n"
                f.write(line_i)
#            raise Exception(line_i)

class WeightDict(dict):
    def __init__(self, arg=[]):
        super(WeightDict, self).__init__(arg)

    def Z(self):
        return sum(list(self.values()))

    def norm(self):
        Z=self.Z()
        for i in self:
            self[i]= self[i]/Z
        return self
    
    def gini(self):
        arr=list(self.values())
        arr.sort()
        arr=np.array(arr)
        index = np.arange(1,arr.shape[0]+1)
        n = arr.shape[0]     
        return ((np.sum((2 * index - n  - 1) * arr)) / (n * np.sum(arr)))

def read_csv(in_path:str):
    if(type(in_path)==tuple):
        X,y=in_path
        return Dataset(X,y)
    if(type(in_path)!=str):
        return in_path
    df=pd.read_csv(in_path,header=None)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    X= preprocessing.RobustScaler().fit_transform(X)
    return Dataset(X,y)

class DFView(object):
    def __init__(self,df):
        self.df=df.round(4)

    def to_csv(self):
        text=",".join(self.df.columns)
        for index, row in self.df.iterrows():
            line_i=",".join([str(c_i) for c_i in row.to_list()])
            text+="\n"+line_i
        return text

    def print(self,dec=4):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
            print(self.df)

def make_df(helper,
            iterable,
            cols,
            offset=None,
            multi=False):
    lines=[]
    if(multi):
        for arg_i in iterable:
            lines+=helper(arg_i)
    else:
        for arg_i in iterable:
            lines.append(helper(arg_i))
    if(offset):
        line_len=max([len(line_i) for line_i in lines])
        for line_i in lines:
            while(len(line_i)<line_len):
                line_i.append(offset)
        cols+=[str(i)for i in range(line_len-len(cols))]
    df=pd.DataFrame.from_records(lines,
                                columns=cols)
    return DFView(df)

def get_class_weights(y):
    params=WeightDict() 
    cats=  list(set(y))
    n_cats= len(cats) 
    params=WeightDict({cat_i:0 for cat_i in cats})
    for y_i in y:
        params[y_i]+=1
    return params.norm()

def data_desc(in_path,first_set=None):
    first_set=set(first_set)
    def helper(in_path):
        name=in_path.split("/")[-1]
        name=name.split(".")[0]
        data=read_arff(in_path,
                       first=(name in first_set))
        return [name,data.gini(),data.n_cats(), 
                data.dim(),len(data)]
    df=make_df(helper=helper,
            iterable=utils.top_files(in_path),
            cols=["data","gini","classes",
#                  "feats",
                  "samples","propor"],
            offset=None,
            multi=False)
    df.print()
#    print(df.to_csv())

def read_arff(in_path:str,first=False):
    if(first):
        print(in_path)
        get_target=target_first
    else:
        print(in_path)
        get_target=target_last
    X,y=[],[]
    with open(in_path) as f:
        for line_i in f:
            if( (not '@' in line_i) and 
                (not '%' in line_i)):
                line_i=line_i.rstrip()
                line_i=line_i.split(",")
                if(len(line_i)>1):
                    x_i,y_i=get_target(line_i)
                    X.append(x_i)
                    y.append(y_i)
        X=[[row[i] for row in X] 
                for i in range(len(X[0]) )]
        X=[preproc(feat_i) for feat_i in X]
        return Dataset(X=np.array(X),
                   y=preproc(y,target=True))

def target_last(line_i):
    return line_i[:-1],line_i[-1]

def target_first(line_i):
    return line_i[1:],line_i[0]

def preproc(feat,target=False):
    if(is_float(feat[0]) and (not target)):
        return [conv(f) for f in feat]
    else:
        cats=list(set(feat))
        cats={cat_i:i for i,cat_i in enumerate(cats)}
        return [cats[f] for f in feat]

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False 

def conv(f):
    if(f=='?'):
        return 0
    else:
        return float(f)

def arff_to_csv(in_path,out_path,first_set=None):
    first_set=set(first_set)
    utils.make_dir(out_path)
    for i,path_i in enumerate(utils.top_files(in_path)):
        id_i=path_i.split("/")[-1].split(".")[0]
        data_i=read_arff(path_i,first=(id_i in first_set))
        data_i.save_csv(f"{out_path}/{id_i}")

if __name__ == '__main__':
    arff_to_csv("AutoML","csv",
              ["madeline","philippine","sylvine"])