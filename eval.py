import numpy as np
from scipy import stats
import base,data,protocol,utils

#@utils.DirFun([("data_path",0),("model_path",1)])
def stat_sig(data_path:str,
             model_path:str,
             protocol_obj:protocol.Protocol,
             clf_type="RF"):
    dataset=data.get_data(data_path)
    exp_io= protocol_obj.get_group(exp_path=model_path)
    rf_results,ne_results=[],[]
    for nescf_ij in exp_io.iter_necscf(dataset):
        nescf_ij.train(clf_type)
        ne_results.append(nescf_ij.eval())     
        rf_results.append(nescf_ij.baseline(dataset,clf_type))
#    print(rf_results)
    pvalue,clf_mean,ne_mean=compute_pvalue(rf_results,ne_results)
    text=f"pvalue:{pvalue:.3f},clf:{clf_mean:.3f},ne:{ne_mean:.3f}"
    print(text)
    return text

def compute_pvalue(clf_results,ne_results):
    clf_acc=[result_i.acc() for result_i in clf_results]
    ne_acc=[result_i.acc() for result_i in ne_results]
    pvalue=stats.ttest_ind(clf_acc,ne_acc,equal_var=False)[1]
    return pvalue,np.mean(clf_acc),np.mean(ne_acc)

@utils.DirFun([("data_path",0),("model_path",1)])
def clf_comp(data_path:str,
             model_path:str,
             protocol_obj:protocol.Protocol,
             clf_single="RF",
             clf_ne="LR"):
    dataset=data.get_data(data_path)
    exp_group= protocol_obj.get_group(exp_path=model_path)
    clf_results,ne_results=[],[]
    for exp_i in exp_group.iter_exp(dataset):
        ne_results.append(exp_i.eval(protocol_obj.alg_params,
                                     clf_type=clf_ne))        
        clf_results.append(exp_i.split.eval(clf_type=clf_single))
    print(f"{clf_single},{acc_stats(clf_results)}")
    print(f"{clf_ne},{acc_stats(ne_results)}")

def acc_stats(results):
    acc=[result_i.acc() for result_i in results]
    return f"mean:{np.mean(acc):.3f},std:{np.std(acc):.3f}"

def indiv_acc(data_path:str,
              model_path:str,
              protocol_obj:protocol.Protocol,
              clf_type="RF"):
    dataset=data.get_data(data_path)
    exp_io= protocol_obj.get_group(exp_path=model_path)
    n_split=protocol_obj.split_gen.n_split

    all_results,mean_acc=[[]],[]
    for k,nescf_ij in enumerate(exp_io.iter_necscf(dataset)):
        nescf_ij.train(clf_type)
        result_k=nescf_ij.eval()
        if((k% n_split)!=0):
            all_results[-1].append(result_k.acc())
        else:
            mean_acc.append(np.mean(all_results[-1]))            
            all_results.append([result_k.acc()])
            print(mean_acc)
    print(mean_acc)

if __name__ == '__main__':
    parser =  utils.get_args(['data','model'],
                             ['n_split','n_iter'])
    args = parser.parse_args()

    prot=protocol.Protocol(io_type=protocol.NNetIO,
                           split_gen=protocol.SplitGenerator(n_split=args.n_split,
                                                             n_iters=args.n_iter))
    r_dict=indiv_acc(data_path=args.data,
                    model_path=args.model,
                    protocol_obj=prot)
    print(r_dict)
#    utils.print_dict(r_dict)