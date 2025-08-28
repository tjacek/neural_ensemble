import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
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
        return self.X.shape[1]
        
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
    
    def selection(self,indices):
        return Dataset(X=self.X[indices],
                       y=self.y[indices])

    def fit_clf(self,train,clf):
        X_train,y_train=self.X[train],self.y[train]
        history=clf.fit(X_train,y_train)
        return clf,history

    def pred(self,test_index,clf):
        if(test_index is None):
            X_test,y_test=self.X,self.y
        else:
            X_test,y_test=self.X[test_index],self.y[test_index]
        y_pred=clf.predict(X_test)
        return Result(y_pred,y_test)

    def eval(self,train_index,test_index,clf,as_result=True):
        clf,history=self.fit_clf(train_index,clf)
        result=self.pred(test_index,clf)
        return result,history

    def binarize(self,k):
        y_k=[ int(k==y_i) for y_i in self.y]  
        return Dataset(X=self.X,
                       y=y_k)
        
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

class Result(object):
    def __init__(self,y_pred,y_true):
        self.y_pred=y_pred
        self.y_true=y_true

    def get_acc(self):
        return accuracy_score(self.y_pred,self.y_true)

    def get_balanced(self):
        return balanced_accuracy_score(self.y_pred,self.y_true)

    def get_metric(self,metric_type):
        if(type(metric_type)==str):
            metric=dispatch_metric(metric_type)
        else:
            raise Exception(f"Arg metric_type should be str is {type(metric_type)}")
        return metric(self.y_pred,self.y_true)

    def report(self):
        print(classification_report(self.y_pred,self.y_true,digits=4))

    def true_pos(self):
        pos= [int(pred_i==true_i) 
                for pred_i,true_i in zip(self.y_pred,self.y_true)]
        return np.array(pos)

    def save(self,out_path):
        y_pair=np.array([self.y_pred,self.y_true])
        np.savez(out_path,y_pair)

class ResultGroup(object):
    def __init__(self,results):
        self.results=results

    def __len__(self):
        return len(self.results)

    def get_metric(self,metric_type):
        return [result_j.get_metric(metric_type) 
                    for result_j in self.results]
    def get_acc(self):
        return [result_j.get_acc() 
                    for result_j in self.results]

    def get_balanced(self):
        return [result_j.get_balanced() 
                    for result_j in self.results]

    def save(self,out_path):
        utils.make_dir(out_path)
        for i,result_i in enumerate(self.results):
            result_i.save(f"{out_path}/{i}")

def read_result(in_path:str):
    if(type(in_path)==Result):
        return in_path
    raw=list(np.load(in_path).values())[0]
    y_pred,y_true=raw[0],raw[1]
    return Result(y_pred=y_pred,
                  y_true=y_true)

def read_result_group(in_path:str):
    results= [ read_result(path_i) 
                 for path_i in utils.top_files(in_path)]
    return ResultGroup(results)

class PartialResults(object):
    def __init__(self,y_true,y_partial):
        self.y_true=y_true
        self.y_partial=y_partial
    
    def __len__(self):
        return self.y_partial.shape[0]

    def vote(self):
        ballot= np.sum(self.y_partial,axis=0)
        return np.argmax(ballot,axis=1)

    def get_metric(self,metric_type="acc"):
        y_pred=self.vote()
        metric=dispatch_metric(metric_type)
        return metric(self.y_true,y_pred)

    def selected_acc(self,subset,metric_type="acc"):
        s_votes=[self.y_partial[i] for i in subset]
        s_ballot= np.sum(s_votes,axis=0)
        s_pred=np.argmax(s_ballot,axis=1)
        metric=dispatch_metric(metric_type)
        return metric(self.y_true,s_pred)

    def indiv(self,metric_type="acc"):
        metric=dispatch_metric(metric_type)
        return [metric(self.y_true,np.argmax(y_i,axis=1)) 
                    for y_i in self.y_partial]

    def save(self,out_path):
        np.savez(out_path,name1=self.y_partial,name2=self.y_true)

class PartialGroup(object):
    def __init__(self,partials):
        self.partials=partials
   
    def n_clfs(self):
        return self.partials[0].y_partial.shape[0]

    def indv_acc(self,metric_type="acc"):
        n_clf= self.n_clfs()
        raw_votes= [ result_i.indiv(metric_type) 
                    for result_i in self.partials]
        raw_votes=np.array(raw_votes)
        return np.mean(raw_votes,axis=0)

def dispatch_metric(metric_type):
    metric_type=metric_type.lower()
    if(metric_type=="acc"):
        return accuracy_score
    if(metric_type=="accuracy"):
        return accuracy_score
    if(metric_type=="balance"):
        return balanced_accuracy_score
    if(metric_type=="balanced accuracy"):
        return balanced_accuracy_score
    raise Exception(f"Unknow metric type:{metric_type}")

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
 
    def by_data(self,sort="acc"):
        names=self.df['data'].unique()
        for name_i in names:
            df_i=self.df[self.df['data']==name_i]
            df_i=df_i.sort_values(by=sort,ascending=False)
            print(df_i)

#    def group(self,sort="acc"):
#        grouped=self.df.groupby(by='data')
#        def helper(df_i):
#            return df_i.sort_values(by=sort)
#            print(df.round(4))
#            return df['dataset'].tolist()
#        self.df= grouped.apply(helper)

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

def from_lines(lines,cols):
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

def csv_desc(in_path):
    for path_i in utils.top_files(in_path):
        data_i=read_csv(path_i)
        print(f"{path_i}-{data_i.n_cats()}")


if __name__ == '__main__':
#    arff_to_csv("AutoML","csv",
#              ["madeline","philippine","sylvine"])
#    data_desc("AutoML",
#        first_set=["madeline","philippine","sylvine"])
    csv_desc("uci_exp/data")