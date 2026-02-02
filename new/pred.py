import numpy as np
import dataset, utils

class ResultDict(dict):
    def clfs(self):
        all_clfs=[]
        for value_i in self.values():    
            all_clfs+= list(value_i.keys())
        all_clfs= list(set(all_clfs))
        all_clfs.sort()
        return all_clfs

    def data(self):
        all_data=list(self.keys())
        all_data.sort()
        return all_data
    
    def __call__(self,clf_type,fun):
        data_dict={}
        for data_i,dict_i in self.items():
            if(clf_type in dict_i):
                data_dict[data_i]=fun(dict_i[clf_type])
        return data_dict

    def get_clf( self,
                 clf_type,
                 metric="acc",
                 split=None):
        
        output=self(clf_type,lambda r:r.get_metric(metric))
        if(split):
            output={ name_i: [ np.mean(v_i)
                for v_i in utils.split_list(value_i,split)] 
                    for name_i,value_i in output.items()}
        return output

    def get_mean_metric(self,clf_type,metric="acc"):
        return self(clf_type,lambda r:np.mean(r.get_metric(metric)))
    
    def compute_metric(self,metric):
        return { data_i:{name_j:result_j.get_metric(metric) 
                       for name_j,result_j in dict_i.items()}
              for data_i,dict_i in self.items()}

    @classmethod
    def read(cls,in_path):
        @utils.DirFun("in_path")
        def helper(in_path):
            output={}
            for path_i in utils.top_files(in_path):
                name_i=path_i.split("/")[-1]
                if(name_i!="splits"):
                    result_path_i=f"{path_i}/results"
                    result_i=dataset.ResultGroup.read(result_path_i)
                    output[name_i]=result_i
            return output
        return cls(helper(in_path))

    def add_partial(self,partial_dict):
        for key_i in self:
            _,_,result_i=partial_dict.best_subset(key_i)
            self[key_i]["TreeEns"]=result_i

    def add_single(self,partial_dict):
        for key_i in self:
            partial_i=partial_dict[key_i].subsets([0])
            result_i=partial_i.to_result()
            self[key_i]["Tree-TabPF"]=result_i

class PartialDict(dict):
    def subsets(self):
        for key_i,partial_i in self.items():
            k,acc,_=self.best_subset(key_i)
            print(f"{key_i},{k},{acc:.4f}")
    
    def best_subset(self,key):
        partial=self[key]
        n_clfs=partial.n_clfs()
        subsets,acc,results=[],[],[]
        for i in range(n_clfs):
            subsets.append(i)
            s_partial_i=partial.subsets(subsets)
            result_i=s_partial_i.to_result()
            acc.append(np.mean(result_i.get_acc()))
            results.append(result_i)
        k=np.argmax(acc)
        return k,acc[k],results[k]

    @classmethod
    def read(cls,in_path):
        @utils.DirFun("in_path")
        def helper(in_path):
            clf_path=f"{in_path}/TreeEnsTabPF/partials"
            result_type=dataset.PartialGroup
            result= result_type.read(clf_path)
            print(in_path)
            return result
        return cls(helper(in_path))

def summary(exp_path,
            latex=True):
    partial_dict=PartialDict.read(exp_path)
    result_dict=ResultDict.read(exp_path)
    result_dict.add_partial(partial_dict)
    result_dict.add_single(partial_dict)
    def df_helper(clf_type):
        acc_dict=result_dict.get_clf(clf_type,metric="acc")
        balance_dict=result_dict.get_clf(clf_type,metric="balance")
        lines=[]
        for data_i in acc_dict:
            line_i=[data_i,clf_type]
            line_i.append(np.mean(acc_dict[data_i]))
            line_i.append(np.mean(balance_dict[data_i]))
            line_i.append(len(acc_dict[data_i]))
            lines.append(line_i)
        return lines
    df=dataset.make_df(helper=df_helper,
                      iterable=result_dict.clfs(),
                      cols=["data","clf","acc","balance","n_splits"],
                      multi=True)
    for df_i in df.by_data("acc"):   
        if(latex):
            to_latex(df_i)
        else:
            print(df_i)

def to_latex(df):
    df=df[["data","clf","acc"]]
    cols=df.columns.to_list()
    def helper(raw):
        if(type(raw)==float):
            return f"{100*raw:.2f}"
        return str(raw)
    for i,row_i in df.iterrows():
        row_i=row_i.to_dict() 
        row_i=[helper(row_i[col_j])
                   for col_j in cols]
        line_i=" & ".join(row_i)
        line_i=f"\\hline {line_i} \\\\"
        print(line_i)

summary("uci/fast_exp/exp")
#cls_type=PartialDict
#result_dict=cls_type.read("test_exp")
#result_dict.subsets()