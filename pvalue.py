import dataset,utils

def metric_dict(in_path,metrics="acc"):
#    if(metrics is None):
#        metrics=["acc","balance"]
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
    output_dict=helper(in_path)
    print(output_dict)



in_path="uci_exp/exp"
metric_dict(in_path,metrics="acc")