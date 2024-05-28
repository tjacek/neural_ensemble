import numpy as np
from sklearn.decomposition import PCA
import base,data,protocol,utils

class DeoractorPCA(protocol.ExpIO):
    def __init__(self,io_type,pca=True,raw=True,both=True):
        self.io_type=io_type
        self.exp_path=io_type.exp_path
        self.n_iters=io_type.n_iters
        self.n_split=io_type.n_split
        self.pca=pca 
        self.raw=raw
        self.both=both

    def get_necscf(self,i,j,path,dataset):
        exp_ij=self.io_type.get_exp(i,j,path,dataset)
        extractor= exp_ij.make_extractor()
        cs=extractor.predict(dataset.X)
        dim=dataset.params["dims"]
        pca_feats = PCA(n_components=dim).fit_transform(dataset.X)
        raw_splits,pca_splits=[],[]
        for cs_i in cs:
            feats_i=np.concatenate([dataset.X,cs_i],axis=1)
            split_i= make_split(feats_i,dataset,exp_ij)
            raw_splits.append(split_i)
            pca_feats_i=np.concatenate([pca_feats,cs_i],axis=1)
            split_i= make_split(pca_feats_i,dataset,exp_ij)
            pca_splits.append(split_i)
        if(self.raw):
            yield "base",base.NECSCF(all_splits=raw_splits)

        if(self.pca):
            yield "pca",base.NECSCF(all_splits=pca_splits)

        if(self.both):
            yield "mixed",base.NECSCF(all_splits=raw_splits+pca_splits)

def make_split(feats_i,dataset,exp_i):
    data_i=data.Dataset(X=feats_i,
                        y=dataset.y,
                        params=dataset.params)
    return base.Split(dataset=data_i,
                      train=exp_i.split.train,
                      test=exp_i.split.test)


@utils.DirFun([("data_path",0),("model_path",1)])
def stat_sig(data_path:str,
             model_path:str,
             protocol_obj:protocol.Protocol,
             clf_type="RF"):
    dataset=data.get_data(data_path)
    exp_io= protocol_obj.get_group(exp_path=model_path)
    exp_io=DeoractorPCA(exp_io)
    ne_results={"base":[],"pca":[],"mixed":[]}
    for type_i,nescf_ij in exp_io.iter_necscf(dataset):
        nescf_ij.train(clf_type)
        print(nescf_ij)
        acc_ij=nescf_ij.eval().acc()
        ne_results[type_i].append(acc_ij)    
    text=""
    for name_i,acc_i in ne_results.items(): 
         text+=f"{name_i}:{np.mean(acc_i):4f}"
    print(text)
    return text

def show_result(ret_dict):
    for data_i,text_i in ret_dict.items():
        new_text=[f'{data_i},{line_j}' for line_j in text_i]
        print('\n'.join(new_text))

if __name__ == '__main__':
    prot=protocol.Protocol(io_type=protocol.NNetIO,
                           split_gen=protocol.SplitGenerator(n_split=10,
                                                             n_iters=10))
    ret_dict=stat_sig(data_path=f"../uci/new",#/cleveland",
                      model_path=f"../10-10/new",#cleveland",
                      protocol_obj=prot)
    print(ret_dict)