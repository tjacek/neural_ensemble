import numpy as np
from sklearn.decomposition import PCA
import data,protocol,utils

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
        pca_feats = PCA(n_components=5).fit(dataset.X)
        for cs_i in cs:
            feats_i=np.concatenate([dataset.X,cs_i],axis=1)
            split_i= make_split(feats_i,dataset,exp_i)
            all_splits.append(split_i)
            pca_feats_i=np.concatenate([pca_feats,cs_i],axis=1)
            split_i= make_split(pca_feats_i,dataset,exp_i)
            all_splits.append(split_i)
        return base.NECSCF(all_splits=all_splits)

def make_split(feats_i,dataset,exp_i):
    data_i=data.Dataset(X=feats_i,
                        y=dataset.y,
                        params=dataset.params)
    return protocol.Split(dataset=data_i,
                           train=exp_i.train,
                            test=exp_i.test)

#@utils.DirFun([("data_path",0),("model_path",1)])
def stat_sig(data_path:str,
             model_path:str,
             protocol_obj:protocol.Protocol,
             clf_type="RF"):
#    protocol_obj.io_type= DeoractorPCA(protocol_obj.io_type)
    dataset=data.get_data(data_path)
    exp_io= protocol_obj.get_group(exp_path=model_path)
    exp_io=DeoractorPCA(exp_io)
    for nescf_ij in exp_io.iter_necscf(dataset):
    	print(nescf_ij)

if __name__ == '__main__':
    prot=protocol.Protocol(io_type=protocol.NNetIO,
                           split_gen=protocol.SplitGenerator(n_split=3,
                                                             n_iters=3))
    hyper_dict=stat_sig(data_path=f"../uci/old/cleveland",
                          model_path=f"../10-10/cleveland",
                          protocol_obj=prot)