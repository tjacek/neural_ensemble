import numpy as np
from sklearn.decomposition import PCA
import base,data,protocol,utils

class DeoractorPCA(protocol.ExpIO):
    def __init__(self,io_type):
        self.io_type=io_type
        self.exp_path=io_type.exp_path
        self.n_iters=io_type.n_iters
        self.n_split=io_type.n_split

    def get_necscf(self,i,j,path,dataset):
        exp_ij=self.io_type.get_exp(i,j,path,dataset)
        extractor= exp_ij.make_extractor()
        cs=extractor.predict(dataset.X)
        dim=dataset.params["dims"]
        pca_feats = PCA(n_components=dim).fit_transform(dataset.X)
        all_splits=[]
        for cs_i in cs:
            feats_i=np.concatenate([dataset.X,cs_i],axis=1)
            split_i= make_split(feats_i,dataset,exp_ij)
            all_splits.append(split_i)
            pca_feats_i=np.concatenate([pca_feats,cs_i],axis=1)
#            raise Exception(pca_feats_i.shape)
            split_i= make_split(pca_feats_i,dataset,exp_ij)
            all_splits.append(split_i)
        return base.NECSCF(all_splits=all_splits)

def make_split(feats_i,dataset,exp_i):
    data_i=data.Dataset(X=feats_i,
                        y=dataset.y,
                        params=dataset.params)
    return base.Split(dataset=data_i,
                      train=exp_i.split.train,
                      test=exp_i.split.test)

#@utils.DirFun([("data_path",0),("model_path",1)])
def stat_sig(data_path:str,
             model_path:str,
             protocol_obj:protocol.Protocol,
             clf_type="RF"):
    dataset=data.get_data(data_path)
    exp_io= protocol_obj.get_group(exp_path=model_path)
    exp_io=DeoractorPCA(exp_io)
    ne_results=[]
    for nescf_ij in exp_io.iter_necscf(dataset):
        print(nescf_ij)
        nescf_ij.train(clf_type)
        ne_results.append(nescf_ij.eval().acc())    
    print(np.mean(ne_results))

if __name__ == '__main__':
    prot=protocol.Protocol(io_type=protocol.NNetIO,
                           split_gen=protocol.SplitGenerator(n_split=10,
                                                             n_iters=10))
    hyper_dict=stat_sig(data_path=f"../uci/old/cleveland",
                          model_path=f"../10-10/cleveland",
                          protocol_obj=prot)