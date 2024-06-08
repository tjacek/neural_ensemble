import utils
utils.silence_warnings()
import base,data,exp,deep
import numpy as np
import json
import base,exp,protocol,utils

#@utils.DirFun([("data_path",0),("hyper_path",1),('out_path',2)])
def train_data(data_path:str,
	           hyper_path:str,
	           out_path:str,
	           protocol_obj:protocol.Protocol,
	           verbose=0):
    print(data_path)
    dataset=data.get_data(data_path)
    hyper_params=read_hyper(hyper_path,protocol_obj.alg_params)
    exp_io= protocol_obj.get_group(exp_path=out_path)
    exp_io.init_dir()
    for i,j,split_i in protocol_obj.split_gen.iter(dataset):
        exp_i=exp.make_exp(split_i,hyper_params)
        exp_i.train(protocol_obj.alg_params,
        	        verbose=verbose)
        exp_io.set(exp_i,i,j,dataset)

def read_hyper(hyper_path,alg_params):
    with open(hyper_path) as out_file:
        hyper_dict = json.loads(out_file.read())
        hyper_dict['alpha']=alg_params.alpha
        return hyper_dict

if __name__ == '__main__':
#    hyper_params={'units_0': 123, 'units_1': 65, 'batch': 0, 'layers': 2}
    parser =  utils.get_args(['data','hyper','model'])
    args = parser.parse_args()

    prot=protocol.Protocol(io_type=protocol.FeatIO,
                           split_gen=protocol.SplitGenerator(n_split=args.n_split,
                                                             n_iters=args.n_iter))
    hyper_dict=train_data(data_path=args.data,#'../uci/cleveland',
    	                  hyper_path=args.hyper,#'../hyper/cleveland',
    	                  out_path=args.model,#"../cleveland",
                          protocol_obj=prot,
                          verbose=0)