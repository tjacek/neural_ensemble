import utils
utils.silence_warnings()
import base,data,exp,deep
import numpy as np
import json
import base,exp,protocol,utils

@utils.DirFun([("data_path",0),("hyper_path",1),('out_path',2)])
def train_data(data_path:str,
	           hyper_path:str,
	           out_path:str,
	           protocol_obj:protocol.Protocol,
	           verbose=0):
    print(in_path)
    dataset=data.get_data(data_path)
    hyper_params=read_hyper(hyper_path)
    exp_io= protocol_obj.get_group(exp_path=out_path)
    exp_io.init_dir()
    for i,j,split_i in protocol_obj.split_gen.iter(dataset):
        exp_i=exp.make_exp(split_i,hyper_params)
        exp_i.train(protocol_obj.alg_params,
        	        verbose=verbose)
        exp_io.set(exp_i,i,j)

def read_hyper(hyper_path:str):
    with open(hyper_path) as out_file:
        return json.loads(out_file.read())

if __name__ == '__main__':
    in_path='../uci' #cleveland'
#    hyper_params={'units_0': 204, 'units_1': 52, 'batch': 0, 'layers': 2}
    hyper_params={'units_0': 123, 'units_1': 65, 'batch': 0, 'layers': 2}
    prot=protocol.Protocol(io_type=protocol.NNetIO,
                           split_gen=protocol.SplitGenerator(n_split=3,
                                                             n_iters=3))
    hyper_dict=train_data(data_path=in_path,
    	                  hyper_path='../hyper',
    	                  out_path="../test2",
                          protocol_obj=prot,
                          verbose=0)