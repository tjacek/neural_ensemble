import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))
import argparse
import numpy as np
import pandas as pd 
import conf,data,learn,plot

def gen_cf(conf ,sep=','):
    df=pd.read_csv(conf['best'])
    data.make_dir(conf['cf'])
    result_dict=plot.read_results(conf['output'])
    for i,row_i in df.iterrows():
        data_i=row_i['dataset']
        alg_i=sep.join ([row_i['ens_type'],row_i['clf_type']])
        results=result_dict[data_i][alg_i]
        acc=[result_j.get_acc() 
              for result_j in results] 
        median=np.argsort(acc)[len(acc)//2]
        cf_i=results[median].get_cf()
        out_i="{}/{}".format(conf['cf'],alg_i)
        np.savetxt(out_i,cf_i)       
        print(f'Saved conf matrix at {out_i}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",type=str,default='conf/base.cfg')
    args = parser.parse_args()
    conf_dict =conf.read_conf(args.conf,'dir')
    gen_cf(conf_dict)