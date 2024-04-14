import numpy as np
from scipy import stats
import base,data,protocol,utils

@utils.DirFun([("data_path",0),("model_path",1)])
def stat_sig(data_path:str,
             model_path:str,
             protocol_obj:protocol.Protocol,
             clf_type="RF"):
    dataset=data.get_data(data_path)
    exp_group= protocol_obj.get_group(exp_path=model_path)
    rf_results,ne_results=[],[]
    for exp_ij in exp_group.iter_exp(dataset):
        ne_results.append(exp_ij.eval(protocol_obj.alg_params))        
        rf_results.append(exp_ij.split.eval(clf_type))
    pvalue,clf_mean,ne_mean=compute_pvalue(rf_results,ne_results)
    print(f"pvalue:{pvalue:.3f},clf:{clf_mean:.3f},ne:{ne_mean:.3f}")    
#def stat_sig(in_path:str,
#		     data_path:str,
#	         alg_params=None,
#	         clf_type="RF"):
#    exp_facade=protocol.read_facade(in_path)
#    dataset=data.get_data(data_path)
#    if(alg_params is None):
#        alg_params=base.AlgParams()
#    rf_results,ne_results=[],[]
#    for exp_i in exp_facade.iter(dataset):
#        print(exp_i)
#        ne_results.append(  exp_i.eval(alg_params))        
#        rf_results.append(exp_i.split.eval(clf_type))
#    pvalue,clf_mean,ne_mean=compute_pvalue(rf_results,ne_results)
#    print(f"pvalue:{pvalue:.3f},clf:{clf_mean:.3f},ne:{ne_mean:.3f}")

def compute_pvalue(clf_results,ne_results):
    clf_acc=[result_i.acc() for result_i in clf_results]
    ne_acc=[result_i.acc() for result_i in ne_results]
    pvalue=stats.ttest_ind(clf_acc,ne_acc,equal_var=False)[1]
    return pvalue,np.mean(clf_acc),np.mean(ne_acc)

#def clf_comp(in_path:str,
#             data_path:str,
#             alg_params=None,
#             clf_single="RF",
#             clf_ne="LR"):
#    exp_facade=protocol.read_facade(in_path)
#    dataset=data.get_data(data_path)
#    if(alg_params is None):
#        alg_params=base.AlgParams()
#    clf_results,ne_results=[],[]
#    for exp_i in exp_facade.iter(dataset):
#        ne_results.append(exp_i.eval(alg_params,
#                                     clf_type=clf_single))        
#        clf_results.append(exp_i.split.eval(clf_type=clf_single))
#    print(f"{clf_single},{acc_stats(clf_results)}")
#    print(f"{clf_ne},{acc_stats(ne_results)}")

def acc_stats(results):
    acc=[result_i.acc() for result_i in results]
    return f"mean:{np.mean(acc):.3f},std:{np.std(acc):.3f}"

if __name__ == '__main__':
    prot=protocol.Protocol(io_type=protocol.NNetIO,
                           split_gen=protocol.SplitGenerator(n_split=3,
                                                             n_iters=3))
    stat_sig(data_path=f"../uci",
             model_path=f"../test2",
             protocol_obj=prot)