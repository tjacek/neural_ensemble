import ens,utils

class ESCFExp(object):
    def __init__(self,ens_types=[ens.Ensemble],
        clf_types=['LR','RF'],ensemble_reader=None):
        if(ensemble_reader is None):
            ensemble_reader=ens.npz_reader
        self.ensemble_reader=ensemble_reader
        self.ens_types=ens_types
        self.clf_types=clf_types

    @utils.dir_fun(False)
    def __call__(self,in_path):
       
        @utils.dir_fun(as_dict=False)
        @utils.unify_cv(dir_path=None)
        def helper(path_i):
            common,binary=self.ensemble_reader(path_i)
            result_dict={}
            for alg_i in self.ens_types:
                for clf_j in self.clf_types:
                    ens_i=alg_i(common,binary,clf_j)
                    result_i=ens_i.evaluate()
                    id_ij=",".join([str(ens_i),clf_j])
                    result_dict[id_ij]=result_i
            return result_dict
        acc=helper(in_path)
        raise Exception(acc)
        print((alg_i,clf_j,acc))

exp=ESCFExp()
exp('test')