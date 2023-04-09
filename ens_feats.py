from multiprocessing import Pool, get_context
import learn,data

class EnsFeatures(object):
    def __init__(self,common,binary):
        self.common=common
        self.binary=binary

    def __call__(self,clf_type='LR'):
        full=[ self.common.concat(binary_i) 
                for binary_i in self.binary]
        results=fit_clf(full,clf_type)
        return learn.voting(results)

    def __str__(self):
        return 'NECSCF'

class ParallelEnsemble(object):
    def __init__(self,common,binary):
        self.common=common
        self.binary=binary

    def __call__(self,clf_type='LR'):
        full=[ self.common.concat(binary_i) 
                for binary_i in self.binary]
        helper= FitClf(clf_type)

        n_models=len(self.binary)
        with get_context("spawn").Pool(n_models) as p:
            results=p.map(helper, self.binary)
        return learn.voting(results)

    def __str__(self):
        return 'NECSCF-parallel'

class FitClf(object):
    def __init__(self, clf):
        self.clf=clf

    def __call__(self,full_i):
        result=learn.fit_clf(full_i,self.clf)
        return result.split()[1]

class BinaryEnsemble(object):
    def __init__(self,common,binary,clf_type=None):
        self.binary=binary 

    def __call__(self,clf_type='LR'):
        results=fit_clf(self.binary,clf_type)
        return learn.voting(results)
      
    def __str__(self):
        return 'binary'

class NoEnsemble(object):        
    def __init__(self,common,binary):
        self.common=common

    def __call__(self,clf_type='LR'):
        result_i=learn.fit_clf(self.common,clf_type)
        result_i=result_i.split()[1]
        return result_i
 
    def __str__(self):
        return 'common'

class PCAEnsemble(object):
    def __init__(self,common,binary):
        self.common=common
        self.binary=binary
        self.pca=None

    def __call__(self,clf_type='LR'):
        if(self.pca is None):
            self.pca=data.transform_data(self.common)
        full=[self.pca.concat(binary_i) 
                for binary_i in self.binary]
        results=fit_clf(full,clf_type)
        return learn.voting(results)      

    def __str__(self):
        return 'pca'

class PCAMixed(object):
    def __init__(self,common,binary):
        self.common=common
        self.binary=binary
        self.pca=None

    def __call__(self,clf_type='LR'):
        if(self.pca is None):
            self.pca=data.transform_data(self.common)
        raw_feats=[self.binary.concat(binary_i) 
                    for binary_i in self.binary]
        pca_feats=[self.pca.concat(binary_i) 
                    for binary_i in self.binary]
        results=fit_clf(raw_feats+pca_feats,clf_type)
        return learn.voting(results)      

    def __str__(self):
        return 'pca-mixed'

class PCANoEnsemble(object):
    def __init__(self,common,binary):
        self.common=common
        self.pca=None

    def __call__(self,clf_type='LR'):
        if(self.pca is None):
            self.pca=data.transform_data(self.common)
        result_i=learn.fit_clf(self.pca,clf_type)
        result_i=result_i.split()[1]
        return learn.voting(results)      

    def __str__(self):
        return 'pca-only'

def fit_clf(full_data,clf_type):
    results=[]
    for full_i in full_data:
        result_i=learn.fit_clf(full_i,clf_type)
        results.append(result_i)
    results=[result_i.split()[1] 
        for result_i in results]    
    return results

def get_ensemble(ens_type):
    if(ens_type=='binary'):
        return BinaryEnsemble
    if(ens_type=='common'):
        return NoEnsemble
    if(ens_type=='pca'):
        return PCAEnsemble
    if(ens_type=='pca-only'):
        return PCANoEnsemble
    if(ens_type=='pca-mixed'):
        return PCAMixed    
    if(ens_type=='para'):
        return ParallelEnsemble
    return EnsFeatures