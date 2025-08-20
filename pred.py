import numpy as np
import argparse
import base,dataset,utils
utils.silence_warnings()

class ResultDict(dict):
    def clfs(self):
        all_clfs=list(self.values())[0].keys()
        all_clfs=list(all_clfs)
        all_clfs.sort()
        return all_clfs

    def data(self):
        all_data=list(self.keys())
        all_data.sort()
        return all_data
    
    def __call__(self,clf_type,fun):
        return { data_i:fun(dict_i[clf_type])
                    for data_i,dict_i in self.items()}

    def get_clf(self,clf_type,metric="acc"):
        return self(clf_type,lambda r:r.get_metric(metric))
    
    def get_mean_metric(self,clf_type,metric="acc"):
        return self(clf_type,lambda r:np.mean(r.get_metric(metric)))
    
    def compute_metric(self,metric):
        return { data_i:{name_j:result_j.get_metric(metric) 
                       for name_j,result_j in dict_i.items()}
              for data_i,dict_i in self.items()}

def get_result_dict(in_path):
    @utils.DirFun(out_arg=None)
    def helper(in_path):
        output={}
        for path_i in utils.top_files(in_path):
            name_i=path_i.split("/")[-1]
            if(name_i!="splits"):
                result_path_i=f"{path_i}/results"
                result_i=dataset.read_result_group(result_path_i)
                output[name_i]=result_i
        return output
    return ResultDict(helper(in_path))

def unify_results(paths):
    indv_dicts=[get_result_dict(path_i) 
                   for path_i in paths]
    raw_dict={}
    for indv_i in indv_dicts:
        raw_dict= raw_dict | indv_i
#    print(raw_dict.keys())
    return ResultDict(raw_dict)

def summary(exp_path):
    result_dict=get_result_dict(exp_path)
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
    print(df.by_data(sort='acc'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default="uci_exp/exp")
    args = parser.parse_args()
    summary(exp_path=args.exp_path)