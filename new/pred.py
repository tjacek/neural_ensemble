import numpy as np
import argparse,os.path
import dataset,plot,utils

class ResultDict(dict):
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
    
    def by_clf(self,clf_type,fun):
        data_dict={}
        for data_i,dict_i in self.items():
            if(clf_type in dict_i):
                data_dict[data_i]=fun(dict_i[clf_type])
        return data_dict
    
    def norm_metric(self,clf_type,fun):
        data_dict={}
        for data_i,dict_i in self.items():
            values= [ fun(value_i) 
                        for value_i in dict_i.values()]
            delta= max(values)-min(values)
            if(clf_type in dict_i):
                metric=fun(dict_i[clf_type])
                metric= (metric-min(values))/delta
                data_dict[data_i]=metric
        return data_dict

    def get_clf( self,
                 clf_type,
                 metric="acc",
                 mean=False,
                 split=None):
        if("norm_" in metric):
            metric=metric.split("_")[1]
            fun=self.metric_fun(metric,mean)
            output=self.norm_metric(clf_type,fun)
        else:
            fun=self.metric_fun(metric,mean)
            output=self.by_clf(clf_type,fun)
        if(split):
            output={ name_i: [ np.mean(v_i)
                for v_i in utils.split_list(value_i,split)] 
                    for name_i,value_i in output.items()}
        return output
    
    @staticmethod
    def metric_fun(metric,
                   mean):
        if(mean):
            return lambda r:np.mean(r.get_metric(metric))
        else:
            return lambda r:r.get_metric(metric)

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
        if(type(in_path)==list):
            output={}
            for path_i in in_path: 
                output= output | helper(path_i)
            return cls(output)
        return cls(helper(in_path))

    def add_partial(self,
                    partial_dict,
                    partial="TreeEns"):
        for key_i in self:
            if(key_i in partial_dict):
                _,_,result_i=partial_dict.best_subset(key_i)
                self[key_i][partial]=result_i

    def add_single(self,
                   partial_dict,
                   single="Tree-TabPF"):
        for key_i in self:
            if(key_i in partial_dict):
                partial_i=partial_dict[key_i].subsets([0])
                result_i=partial_i.to_result()
                self[key_i][single]=result_i

    def drop(self,clf):
        for dict_i in self.values():
            if(clf in dict_i):
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
    def read( cls,
              in_path,
              clf_path="TreeEnsTabPF"):
        @utils.DirFun("in_path")
        def helper(in_path):
            full_path=f"{in_path}/{clf_path}/partials"
            if not os.path.exists(full_path):
                return None
            result_type=dataset.PartialGroup
            result= result_type.read(full_path)
            print(in_path)
            return result
        if(type(in_path)==list):
            output={}
            for path_i in in_path: 
                output= output | helper(path_i)
            return cls(output)
        return cls(helper(in_path))

def get_results(exp_path):
#    part_names=["TreeEnsTabPF","TreeEnsTabPFN"]
    part_names=["TreeEnsTabPFN"]
    result_dict=ResultDict.read(exp_path)
    all_partial={ name_i:PartialDict.read(exp_path,
                                          name_i) 
                    for name_i in  part_names}
    for name_i,dict_i in all_partial.items():
        partial=name_i#f"TreeEns-{name_i}"
        result_dict.add_partial(dict_i,
                                partial)
        single=f"Tree-{name_i}"
        result_dict.add_single(dict_i,
                               single)
    result_dict.drop("TreeTabPF(optim)")
    return result_dict 

def summary(exp_path,
            metric_types=None,
            latex=True):
    if(metric_types is None):
        metric_types=["acc","norm_acc"]
    result_dict=get_results(exp_path)
    def df_helper(clf_type):
        metrics=[result_dict.get_clf(clf_type,
                                     metric=metric_i,
                                     mean=True)
                        for metric_i in metric_types]
        lines=[]
        for data_i in result_dict:
            line_i=[data_i,clf_type]
            for metric_j in metrics:
                if(data_i in metric_j):
                    line_i.append(metric_j[data_i])
                else:
                    line_i.append(-1)
            lines.append(line_i)
        return lines
    cols=["data","clf"]+metric_types#+["n_splits"]
    df=dataset.make_df(helper=df_helper,
                      iterable=result_dict.clfs(),
                      cols=cols,
                      multi=True)
    for df_i in df.by_data("acc"):   
        if(latex):
            to_latex(df_i)
        else:
            print(df_i)
    mean_acc(df)

def mean_acc(df):
    for df_i in df.by_data(col="clf"):
        clf_i=df_i["clf"].tolist()[0]
        acc_i=df_i["norm_acc"].tolist()
        print(f"{clf_i}:{np.mean(acc_i):.4f}")

def to_latex(df):
    df=df[["data","clf","acc","norm_acc"]]
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
    result_dict.drop("TreeEnsTabPF")
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
            x_clf="TabPFN",
            y_clf="TreeEnsTabPFN",
            metric="norm_acc",
            title="AutoML",
            text=False):
    result_dict=get_results(exp_path)
    x_dict=result_dict.get_clf(x_clf,
                               metric=metric,
                               mean=True)
    y_dict=result_dict.get_clf(y_clf,
                               metric=metric,
                               mean=True)
    print(x_dict.keys())
    print(y_dict.keys())
    plot.dict_plot( x_dict,
                    y_dict,
                    xlabel=f"{x_clf}({metric})",
                    ylabel=f"{y_clf}({metric})",
                    text=text,
                    title=title)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=["../binary/hard/exp/",
                                                     "../binary/fast/exp"])
    parser.add_argument('--box', action='store_true')
    parser.add_argument('--xy', action='store_true')
    args = parser.parse_args()
    summary(args.path)
    if(args.box):
        box_plot(args.path)
    if(args.xy):
        xy_plot(["../binary/hard/exp/",
                 "../binary/fast/exp/"])