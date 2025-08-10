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
                acc_j=np.mean(result_j.get_acc())
                balance_j=np.mean(result_j.get_metric("balance"))
                output.append((dir_j.clf_type,acc_j,balance_j))
        return output
    output=helper(exp_path)
    def df_helper(tuple_i):
        name_i,other=tuple_i
        lines=[]
        for tuple_j in other:
            lines.append([name_i]+list(tuple_j))
        return lines
    df=dataset.make_df(helper=df_helper,
                    iterable=output.items(),
                    cols=["data","clf","acc","balance"],
                    multi=True)
    df.print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default="uci_exp/exp")
    args = parser.parse_args()
    summary(exp_path=args.exp_path)