import argparse
import pandas as pd
import tools

def make_summary(dir_path,out_path):
    paths=tools.get_dirs(dir_path)
    for path_i in paths:
        name_i=path_i.split('/')[-1]
        result_i=f'{path_i}/results'
        df=pd.read_csv(result_i) 
        df=df.sort_values(by="acc_mean",ascending=False)
        df_pvalue=pd.read_csv(f'{path_i}/pvalue.txt') 
        with open(out_path,"a") as f:
            f.write(f'{name_i}\n')
            f.write(df.to_csv())
            f.write(df_pvalue.to_csv())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default='../../cl/out')
    parser.add_argument("--out", type=str, default='../../cl/out/summary.txt')
    args = parser.parse_args()
    make_summary(args.dir,args.out)