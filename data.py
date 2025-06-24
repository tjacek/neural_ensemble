import numpy as np
import pandas as pd
import utils

class Dataset(object):
    def __init__(self,X,y=None):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[1]
        
    def n_cats(self):
        return int(max(self.y))+1

    def gini(self):
        return self.weight_dict().gini()

    def weight_dict(self):
        return get_class_weights(self.y)

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
#    def size_dict(self):
#        d={ i:(1.0/w_i) for i,w_i in self.items()}
#        return  WeightDict(d).norm()

class DFView(object):
    def __init__(self,df):
        self.df=df.round(4)

    def to_csv(self):
        text=",".join(self.df.columns)
        for index, row in self.df.iterrows():
            line_i=",".join([str(c_i) for c_i in row.to_list()])
            text+="\n"+line_i
        return text

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
    n_cats= len(cats) #int(max(y))+1
    params=WeightDict({cat_i:0 for cat_i in cats})
    for y_i in y:
        params[y_i]+=1
    print(params)
    return params.norm()

def data_desc(in_path):
    def helper(in_path):
        name=in_path.split("/")[-1]
        data=read_arff(in_path)
        raise Exception(name)
    df=make_df(helper=helper,
            iterable=utils.top_files(in_path),
            cols=["data","gini"],
            offset=None,
            multi=False)

def read_arff(in_path:str):
    X,y=[],[]
    with open(in_path) as f:
        for line_i in f:
            if( (not '@' in line_i) and 
                (not '%' in line_i)):
                line_i=line_i.rstrip()
                line_i=line_i.split(",")
                if(len(line_i)>1):
                    y.append(line_i[-1])
#                X.append([float(cord_j)  for cord_j in line_i[:-1]])
                    X.append(line_i[:-1])
        X=[[row[i] for row in X] 
                for i in range(len(X[0]) )]
        return Dataset(X=np.array(X),
                   y=y)

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False 

if __name__ == '__main__':
    data_desc("AutoML")
#    data=read_arff("AutoML/yeast.arff")
#    w=(data.weight_dict())
#    print(w.gini())