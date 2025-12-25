import base


class TreeFactory(NeuralClfFactory):
    def __init__( self,
                  feature_params=None):
        if(feature_params is None):
            feature_params={'feat_type': 'info', 
                            'n_feats': 20}
        self.feature_params=feature_params

    def __call__(self):
        extractor_factory=FeatureExtactorFactory(**self.feature_params)
        return TreeMLP(params=self.params,
                       hyper_params=self.hyper_params,
                       extractor_factory=extractor_factory)
    
    def get_info(self):
        return {"clf_type":"TREE",
                "feature_params":self.feature_params}