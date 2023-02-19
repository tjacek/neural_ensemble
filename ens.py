import numpy as np
import json,gzip
import learn,data

class Ensemble(object):
    def __init__(self,common,binary,clf_type=None):
        self.common=common
        self.binary=binary 
        self.full=None
        self.clf_type=clf_type

    def evaluate(self,as_votes=False):
        if(self.full is None):
            self.full=[ self.common.concat(binary_i) 
                for binary_i in self.binary]
        print(len(self.full))
        results=[]
        for full_i in self.full:
            result_i=learn.fit_clf(full_i,self.clf_type)
            results.append(result_i)
        results=[result_i.split()[1] 
            for result_i in results]
        if(as_votes):
            return learn.Votes(results)
        return learn.voting(results)

def gzip_reader(in_path):
    with gzip.open(in_path, 'r') as f:        
        json_bytes = f.read()                      
        json_str = json_bytes.decode('utf-8')           
        raw_dict = json.loads(json_str)
        common=data.DataDict(raw_dict['common'])
        binary=[ data.DataDict(binary_i)
            for binary_i in raw_dict['binary']]
        return Ensemble(common,binary,self.clf_type) 

def gzip_writer(ens,out_path):
    raw_dict={'common':ens.save(),
        'binary':[binary_i.save() 
            for binary_i in ens.binary]}
    with gzip.open(out_path, 'wb') as f:
        json_str = json.dumps(raw_dict) + "\n"          
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)

def npz_reader(in_path):
    feat_path=f'{in_path}/feats.npz'
    name_path=f'{in_path}/names'
    raw_feats=np.load(feat_path)
    common=raw_feats['common']
    binary=[]
    for key_i in raw_feats:
        if(key_i!='common'):
            binary.append(raw_feats[key_i])
    with open(name_path, 'r') as f:
        names= json.load(f)
    common=np_to_dict(names,common)
    binary=[np_to_dict(names,binary_i) 
            for binary_i in binary]
    return common,binary

def npz_writer(ens,out_path):
    data.make_dir(out_path)
    names=ens.common.names()
    feats=[('common',ens.common)]
    feats+=[(f'binary_{i}',binary_i) 
         for i,binary_i in enumerate(ens.binary)]
    arr_dict={}
    for key_i,feat_i in feats:
        x_i=[feat_i[name_j] for name_j in names]
        arr_dict[key_i]= x_i
    names.save(f'{out_path}/names')
    np.savez_compressed(f'{out_path}/feats',**arr_dict)

def np_to_dict(names,arr):
    raw_dict= {name_i:arr_i for name_i,arr_i in zip(names,arr)}
    return data.DataDict(raw_dict)

#class RawBinary(object):
#    def __init__(self,clf_type=None):
#        self.clf_type=clf_type

#    def __call__(self,in_path):
#        binary_path=f'{in_path}/binary'
#        binary=data.read_data_group(binary_path)
#        return Ensemble(binary,binary,self.clf_type)

if __name__ == "__main__":
    in_path='imb/wall-following/0/feats/0'
 #   ens_factory=EnsembleFactory()
 #   ens_i=ens_factory(in_path)
 #   ens_i.as_gzip('test.gzip')
    s=GzipFactory()('test.gzip')
    print(type(s))