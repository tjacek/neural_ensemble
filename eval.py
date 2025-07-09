import numpy as np
import argparse
import base,utils

def summary(exp_path):
    @utils.DirFun("exp_path",None)
    def helper(exp_path):
        clf_dirs=[ base.get_dir_path(path_i,clf_type=None)
                    for path_i in utils.top_files(exp_path)
                       if(not "splits" in path_i)]
        acc=[ dir_i.read_results().get_acc() for dir_i in clf_dirs]
        return np.mean(acc)
    output=helper(exp_path)
    print(output)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default="bad_exp/exp")
    args = parser.parse_args()
    summary(exp_path=args.exp_path)