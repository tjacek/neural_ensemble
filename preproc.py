import utils
utils.silence_warnings()
import numpy as np
import argparse
import base

def make_splits(data_path,
                out_path,
                n_splits=10,
                n_repeats=10):
    utils.make_dir(out_path)
    @utils.DirFun({"in_path":0,"out_path":1})
    def helper(in_path,out_path):
        utils.make_dir(out_path)
        data_split=base.get_splits(data_path=in_path,
                                   n_splits=n_splits,
                                   n_repeats=n_repeats)
        split_path=f"{out_path}/splits"
        utils.make_dir(split_path)
        for i,split_i in enumerate(data_split.splits):
            split_i.save(f"{split_path}/{i}")
    helper(data_path,out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=str, default="bad_exp")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--exp", type=str, default="exp")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=10)
    args = parser.parse_args()
    make_splits(data_path=f"{args.main}/{args.data}",
                out_path=f"{args.main}/{args.exp}",
                n_splits=args.n_splits,
                n_repeats=args.n_repeats)