import utils
utils.silence_warnings()
import base,data,exp,deep
import numpy as np
from scipy import stats
import protocol

def train_data(dataset,protocol_obj,alg_params,hyper_params,
	           out_path,verbose=0):
    for split_i in protocol_obj.iter(dataset):
        exp_i=exp.make_exp(split_i,hyper_params)
        exp_i.train(alg_params,
        	        verbose=verbose)
        protocol_obj.add_exp(exp_i)
    protocol_obj.exp_group.save(out_path)

def stat_sig(dataset,protocol_obj,alg_params,hyper_params,
	         clf_type="RF",
	         verbose=0,
	         out_path=None):
    if(out_path):
        utils.make_dir(out_path)	
    rf_results,ne_results=[],[]
    for i,split_i in enumerate(protocol_obj.iter(dataset)):
        exp_i=exp.make_exp(split_i,hyper_params)
        exp_i.train(alg_params,
        	        verbose=0)
        ne_results.append(  exp_i.eval(alg_params))        
        rf_results.append(exp_i.split.eval(clf_type))
        if(out_path):
            exp_i.save(f'{out_path}/{i}')	
    pvalue,clf_mean,ne_mean=compute_pvalue(rf_results,ne_results)
    print(f"{pvalue:.3f},{clf_mean:.3f},{ne_mean:.3f}")

def compute_pvalue(clf_results,ne_results):
    clf_acc=[result_i.acc() for result_i in clf_results]
    ne_acc=[result_i.acc() for result_i in ne_results]
    pvalue=stats.ttest_ind(clf_acc,ne_acc,equal_var=False)[1]
    return pvalue,np.mean(clf_acc),np.mean(ne_acc)

if __name__ == '__main__':
    in_path='../uci/cleveland'
    dataset=data.get_data(in_path)
#    hyper_params={'units_0': 204, 'units_1': 52, 'batch': 0, 'layers': 2}
    hyper_params={'units_0': 123, 'units_1': 65, 'batch': 0, 'layers': 2}
    prot=protocol.BasicProtocol(n_split=3,
    	                        n_iters=3)
    exp_group=protocol.read_basic("new-3-3",in_path)
    print(exp_group.eval(base.AlgParams()))
#    hyper_dict=train_data(dataset=dataset,
#                          protocol_obj=prot,#base.Protocol(n_split=3,n_iters=3),
#                          alg_params=base.AlgParams(),
#                          hyper_params=hyper_params,
#                          verbose=0,
#                          out_path="new-3-3")