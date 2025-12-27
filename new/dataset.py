import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

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

    def fit_clf(self,train,clf):
        X_train,y_train=self.X[train],self.y[train]
        history=clf.fit(X_train,y_train)
        return clf,history

    def eval(self,train_index,test_index,clf,as_result=True):
        clf,history=self.fit_clf(train_index,clf)
        result=self.pred(test_index,clf)
        return result,history
    
    def pred(self,test_index,clf):
        if(test_index is None):
            X_test,y_test=self.X,self.y
        else:
            X_test,y_test=self.X[test_index],self.y[test_index]
        y_pred=clf.predict(X_test)
        return Result(y_pred,y_test)

    def params_dict(self):
        return {"n_cats":self.n_cats(),"dims":(self.dim(),)}

    def range(self):
        return [ (np.amin(x_i),np.amax(x_i))
                    for x_i in self.X.T]

    def weight_dict(self):
        cats=  list(set(self.y))
        n_cats= len(cats) 
        params={cat_i:0 for cat_i in cats}
        for y_i in self.y:
            params[y_i]+=1
        params={key_i:float(value_i) 
                   for key_i,value_i in params.items()}
        return params

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

    @classmethod
    def read(cls,in_path:str):
        if(type(in_path)==Result):
            return in_path
        raw=list(np.load(in_path).values())[0]
        y_pred,y_true=raw[0],raw[1]
        return Result(y_pred=y_pred,
                      y_true=y_true)

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

    @classmethod
    def read(cls,in_path:str):
        results= [ Result.read(path_i) 
                 for path_i in utils.top_files(in_path)]
        return ResultGroup(results)

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
    if(metric_type=="f1"):
        return f1_score
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
            yield df_i

    def best(self,sort_by="acc"):
        grouped=self.df.groupby(by='data')
        def helper(df_i):
            df_i=df_i.sort_values(by=sort_by,ascending=False)
            return df_i.iloc[0]
        return grouped.apply(helper)

    def get_dict(self,x,y):
        return dict(zip(self.df[x].tolist(),
                        self.df[y].tolist()))

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

if __name__ == '__main__':
    data=read_csv("wine-quality-red")
    print(data.range())