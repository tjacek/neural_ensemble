import dataset,utils

def pvalue_matrix(in_path,clf_type="RF",metric="acc"):
    result_dict=get_result_dict(in_path)
    metric_dict=compute_metric(result_dict,metric)
    all_clfs=list(metric_dict.values())[0].keys()
    other_clfs=[clf_i for clf_i in all_clfs
                    if(clf_i!=clf_type) ]
    for data_i in metric_dict:
    	for clf_j in other_clfs:
            print(data_i,clf_j)

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
    return helper(in_path)

def compute_metric(result_dict,metric):
    return { data_i:{name_j:result_j.get_metric(metric) 
                       for name_j,result_j in dict_i.items()}
              for data_i,dict_i in result_dict.items()}

in_path="uci_exp/exp"
pvalue_matrix(in_path,metric="acc")