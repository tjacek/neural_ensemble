import utils
utils.silence_warnings()
import base,data,exp,deep
import numpy as np
import base,protocol,utils

@utils.dir_fun
def train_data(in_path:str,
	           out_path:str,
	           protocol_obj:base.Protocol,
	           alg_params:base.AlgParams,
	           hyper_params:dict,
	           verbose=0):
    print(in_path)
    dataset=data.get_data(in_path)
#    protocol_obj.init_exp_group()
    exp_facade=protocol.ExpFacade(exp_path=out_path,
                                  n_split=protocol_obj.n_split,
                                  n_iters=protocol_obj.n_iters)
    exp_facade.init_dir()
    for i,j,split_i in protocol_obj.iter(dataset):
        exp_i=exp.make_exp(split_i,hyper_params)
        exp_i.train(alg_params,
        	        verbose=verbose)
        exp_facade.set(exp_i,i,j)

if __name__ == '__main__':
    in_path='../uci' #cleveland'
#    hyper_params={'units_0': 204, 'units_1': 52, 'batch': 0, 'layers': 2}
    hyper_params={'units_0': 123, 'units_1': 65, 'batch': 0, 'layers': 2}
    prot=protocol.BasicProtocol(n_split=3,
    	                        n_iters=3)
    alg_params=base.AlgParams()

    hyper_dict=train_data(in_path=in_path,
    	                  out_path="../test",
                          protocol_obj=prot,
                          alg_params=base.AlgParams(),
                          hyper_params=hyper_params,
                          verbose=0)