import numpy as np
import argparse
import base,dataset,utils

def summary(exp_path):
    @utils.DirFun("exp_path",None)
    def helper(exp_path):
        output=[]
        for path_i in utils.top_files(exp_path):
            if(not "splits" in path_i):
                dir_j=base.get_dir_path(path_i,clf_type=None)
                result_j=dir_j.read_results()
                acc_j=result_j.get_acc()
                output.append((dir_j.clf_type,np.mean(acc_j)))
        return output
    output=helper(exp_path)
    def df_helper(tuple_i):
        name_i,other=tuple_i
        lines=[]
        for clf_j,acc_j in other:
            lines.append([name_i,clf_j,acc_j])
        return lines
    df=dataset.make_df(helper=df_helper,
                    iterable=output.items(),
                    cols=["data","clf","acc"],
                    multi=True)
    df.print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default="bad_exp/exp")
    args = parser.parse_args()
    summary(exp_path=args.exp_path)