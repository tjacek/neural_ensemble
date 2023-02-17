import numpy as np

#import protocols,utils,ecscf,ens
import ens,utils,data,learn
import sys

class ESCFExp(object):
    def __init__(self,ensemble_factory=None):
        if(ensemble_factory is None):
            ensemble_factory=escf.EnsembleFactory()
        self.ensemble_factory=ensemble_factory

    @utils.dir_fun(False)
    @utils.unify_cv(dir_path=None)
    def __call__(self,path_i):
        ens_i=self.ensemble_factory(path_i)
        result_i=ens_i.evaluate()
        return result_i

@utils.dir_fun(False)
def check_alg(in_path,clf=None):
    if(clf is None):
        clf=ensemble.RandomForestClassifier()
    results=[]
    feats_path=f'{in_path}/feats'
    for path_i in data.top_files(feats_path):
        common_path_i=f'{path_i}/common'
        data_i= data.read_data(common_path_i)
        result_i=learn.fit_lr(data_i,clf_i=clf)
        results.append(result_i)
    full_results=learn.unify_results(results)
    print(in_path)
    return full_results.get_acc()

def multi_exp(in_path,exp=None):
    if(exp is None):
        exp=basic_exp
    dec_fun=utils.dir_fun(as_dict=True)(exp)
    dict_output=dec_fun(in_path)
    all_lines=[]
    for name_i,lines_i in dict_output.items():
        for line_j in lines_i:
            all_lines.append(f'{name_i},{line_j}')
    print('\n'.join(all_lines))

class BasicExp(object):
    def __init__(self,escf_algs=['LR','RF'],algs=['LR','RF']):
        self.escf_algs=get_escf_algs(escf_algs)
        self.algs=get_algs(algs)

    def __call__(self,in_path):
        lines=[]
        for type_i,ens_i in self.escf_algs.items():
            print(type_i)
            line_i=[f'ESCF,{type_i}']+stats(ens_i(in_path))
            lines.append(','.join(line_i))
        for type_i,clf_i in self.algs.items():
            print(type_i)
            acc_i=check_alg(in_path,clf_i)
            stats_i=','.join(stats(acc_i))
            lines.append(f'{type_i},{stats_i}')	
        print('\n'.join(lines))
        return lines

def stats(acc):
    return [f'{fun_i(acc):.4f}' 
        for fun_i in [np.mean,np.std]]

def get_algs(names):
    alg={}
    for name_i in names:
        alg[name_i]=learn.get_clf(name_i)
    return alg

def get_escf_algs(names):
    alg={}
    for name_i in names:
        raw_i=name_i.split('_')
        clf_i=learn.get_clf(raw_i[0])
        ens_factory=ens.GzipFactory
        alg[name_i]=ESCFExp(ens_factory(clf_i))
    return alg

if __name__ == "__main__":
    if(len(sys.argv)>1):
        data_dir= sys.argv[1]
    else:
#        data_dir='../positional_voting/ECSCF/imb_gzip'
        data_dir=['../uci_gzip/wine']
    print(data_dir)
    basic_exp=BasicExp(['LR','RF'],algs=[])#'Bag','Grad',
#                       ['LR','RF'])#,'Bag','Grad'])
    #lines=basic_exp('imb/wine-quality-red')
    multi_exp(data_dir,basic_exp)
    #print(lines)