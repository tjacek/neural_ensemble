import numpy as np
import argparse
import dataset,plot,utils

class ResultDict(dict):
    PARTIAL="TreeEns"
    SINGLE="Tree-TabPFN"
    SPLITS_DIR="splits"
    RESULT_DIR="results"
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
                 mean=False,
                 split=None):
        if(mean):
            fun=lambda r:np.mean(r.get_metric(metric))
        else:
            fun=lambda r:r.get_metric(metric)
        output=self(clf_type,fun)
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
                if(name_i!=cls.SPLITS_DIR):
                    result_path_i=f"{path_i}/{cls.RESULT_DIR}"
                    result_i=dataset.ResultGroup.read(result_path_i)
                    output[name_i]=result_i
            return output
        return cls(helper(in_path))

    def add_partial(self,partial_dict):
        for key_i in self:
            _,_,result_i=partial_dict.best_subset(key_i)
            self[key_i][self.PARTIAL]=result_i

    def add_single(self,partial_dict):
        for key_i in self:
            partial_i=partial_dict[key_i].subsets([0])
            result_i=partial_i.to_result()
            self[key_i][self.SINGLE]=result_i

    def drop(self,clf):
        for dict_i in self.values():
            del dict_i[clf]

    def clfs_names(self):
        return { name_i:dict_i.keys()
                 for name_i,dict_i in self.items()}

    def common_clfs(self):
        indv_clfs=list(self.clfs_names().values())
        common=set(indv_clfs[0])
        for clf_i in indv_clfs[1:]:
            common=common.intersection(set(clf_i))
        return list(common)

class PartialDict(dict):
    CLF_PATH="TreeEnsTabPFN/partials"
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
            clf_path=f"{in_path}/{cls.CLF_PATH}"
            result_type=dataset.PartialGroup
            result= result_type.read(clf_path)
            print(in_path)
            return result
        return cls(helper(in_path))

def get_results(exp_path):
    partial_dict=PartialDict.read(exp_path)
    result_dict=ResultDict.read(exp_path)
    result_dict.add_partial(partial_dict)
    result_dict.add_single(partial_dict)
    return result_dict

def summary(exp_path,
            latex=True):
    result_dict=get_results(exp_path)
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

def box_plot(exp_path,split_size=None):
    result_dict=get_results(exp_path)
    result_dict.drop("TreeEnsTabPFN")
    data=list(result_dict.keys())
    if(split_size):
        splits=utils.split_list(data,split_size)
    else:
        splits=[data]
    for split_i in splits:
        plot.plot_box(result_dict,
                      data=split_i,
                      clf_types=result_dict.common_clfs())

def xy_plot(exp_path,
            x_clf="GRAD",
            y_clf="TreeEns",
            metric="accuracy",
            title="UCI"):
    result_dict=get_results(exp_path)
    x_dict=result_dict.get_clf(x_clf,mean=True)
    y_dict=result_dict.get_clf(y_clf,mean=True)
    plot.dict_plot( x_dict,
                    y_dict,
                    xlabel=f"{x_clf}({metric})",
                    ylabel=f"{y_clf}({metric})",
                    text=False,
                    title=title)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="multi/exp")
    parser.add_argument('--box', action='store_true')
    parser.add_argument('--xy', action='store_true')
    args = parser.parse_args()
    summary(args.path)
    if(args.box):
        box_plot(args.path)
    if(args.xy):
        xy_plot(args.path)