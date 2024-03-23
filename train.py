import utils
utils.silence_warnings()
import base,data,exp,deep
import numpy as np
from scipy import stats
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
    protocol_obj.init_exp_group()
    for split_i in protocol_obj.iter(dataset):
        exp_i=exp.make_exp(split_i,hyper_params)
        exp_i.train(alg_params,
        	        verbose=verbose)
        protocol_obj.add_exp(exp_i)
    protocol_obj.exp_group.save(out_path)

def stat_sig(exp_group,alg_params,clf_type):
    rf_results,ne_results=[],[]
    for exp_i in exp_group.iter():
        ne_results.append(  exp_i.eval(alg_params))        
        rf_results.append(exp_i.split.eval(clf_type))
    pvalue,clf_mean,ne_mean=compute_pvalue(rf_results,ne_results)
    print(f"pvalue:{pvalue:.3f},clf:{clf_mean:.3f},ne:{ne_mean:.3f}")

def compute_pvalue(clf_results,ne_results):
    clf_acc=[result_i.acc() for result_i in clf_results]
    ne_acc=[result_i.acc() for result_i in ne_results]
    pvalue=stats.ttest_ind(clf_acc,ne_acc,equal_var=False)[1]
    return pvalue,np.mean(clf_acc),np.mean(ne_acc)

def clf_comp(exp_group,alg_params,clf_single="RF",clf_ne="LR"):
    clf_results,ne_results=[],[]
    for exp_i in exp_group.iter():
        ne_results.append(exp_i.eval(alg_params,
        	                         clf_type=clf_single))        
        clf_results.append(exp_i.split.eval(clf_type=clf_single))
    print(f"{clf_single},{acc_stats(clf_results)}")
    print(f"{clf_ne},{acc_stats(ne_results)}")

def acc_stats(results):
    acc=[result_i.acc() for result_i in results]
    return f"mean:{np.mean(acc):.3f},std:{np.std(acc):.3f}"


if __name__ == '__main__':
    in_path='../uci' #cleveland'
   
#    hyper_params={'units_0': 204, 'units_1': 52, 'batch': 0, 'layers': 2}
    hyper_params={'units_0': 123, 'units_1': 65, 'batch': 0, 'layers': 2}
    prot=protocol.BasicProtocol(n_split=3,
    	                        n_iters=3)
    alg_params=base.AlgParams()
#    exp_group=protocol.read_basic("new-3-3",in_path)
#    clf_comp(exp_group,alg_params,"RF")

    hyper_dict=train_data(in_path=in_path,
    	                  out_path="../test",
                          protocol_obj=prot,
                          alg_params=base.AlgParams(),
                          hyper_params=hyper_params,
                          verbose=0)