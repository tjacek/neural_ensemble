import learn

class EnsFeatures(object):
    def __init__(self,common,binary):
        self.common=common
        self.binary=binary

    def __call__(self,clf_type='LR'):
        full=[ self.common.concat(binary_i) 
                for binary_i in self.binary]
        results=[]
        for full_i in full:
            result_i=learn.fit_clf(full_i,clf_type)
            results.append(result_i)
        results=[result_i.split()[1] 
            for result_i in results]
        return learn.voting(results)

    def __str__(self):
        return 'NECSCF'

class BinaryEnsemble(object):
    def __init__(self,common,binary,clf_type=None):
        self.binary=binary 

    def __call__(self,clf_type='LR'):
        results=[]
        for binary_i in self.binary:
            result_i=learn.fit_clf(binary_i,clf_type)
            results.append(result_i.split()[1])
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

def get_ensemble(ens_type):
    if(ens_type=='binary'):
        return BinaryEnsemble
    if(ens_type=='common'):
        return NoEnsemble
    return EnsFeatures