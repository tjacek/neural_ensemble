import data,ens_feats,learn

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
        return ens_feats.gen_result(full,clf_type)

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
        raw_feats=[self.common.concat(binary_i) 
                    for binary_i in self.binary]
        pca_feats=[self.pca.concat(binary_i) 
                    for binary_i in self.binary]
        return ens_feats.gen_result(raw_feats+pca_feats,clf_type)

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
        return result_i      

    def __str__(self):
        return 'pca-only'