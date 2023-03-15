import argparse
import numpy as np
import pandas as pd 
import conf,data,learn

def gen_cf(conf):
    df=pd.read_csv(conf['best'])
    data.make_dir(conf['cf'])
    for i,row_i in df.iterrows():
        dir_i="{}_{}".format(row_i['ens_type'],
        	row_i['clf_type'])
        in_i="{}/{}/{}".format(conf['output'],
        	row_i['dataset'],dir_i)
        out_i="{}/{}_{}".format(conf['cf'],
        	row_i['dataset'],dir_i)
        data.make_dir(out_i)    
        for j,path_j in enumerate(data.top_files(in_i)):
            result_j=learn.read_result(path_j)
            cf_j=result_j.get_cf()
            np.savetxt(f'{out_i}/{j}',cf_j)
        print(f'Saved conf matrix at {out_i}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",type=str,default='conf/base.cfg')
    args = parser.parse_args()
    conf_dict=conf.read_test(args.conf)
    gen_cf(conf_dict)