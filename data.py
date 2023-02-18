import numpy as np
from sklearn import preprocessing
import random
import json,random,os.path

class DataDict(dict):
    def __init__(self, arg=[]):
        super(DataDict, self).__init__(arg)

    def names(self):
        keys=list(self.keys())
        keys.sort()
        all_names=[]
        for name_i in keys:
            if(type(name_i)==Name):
                all_names.append(name_i)
            else:
                all_names.append(Name(name_i))
        return NameList(all_names)

    def dim(self):
        return list(self.values())[0].shape[0]

    def norm(self):
        names=list(self.keys())
        X=np.array([self[name_i] for name_i in names])
        new_X=preprocessing.scale(X)
        for i,name_i in enumerate(names):
            self[name_i]=new_X[i]

    def convert(self):
        for name_i,x_i in self.items():
            self[name_i]=x_i.astype(float)   

    def as_dataset(self):
        names=self.names()
        return self.get_X(names),self.get_labels(names),names

    def get_X(self,names=None):
        if(names is None):
            names=self.names()
        return np.array([self[name_i] for name_i in names])

    def get_labels(self,names=None):
        if(names is None):
            names=self.names()
        return [ name_i.get_cat() for name_i in names]

    def n_cats(self):
        return np.argmax(self.get_labels())

    def split(self,selector=None,shuffle=True):
        if(selector is None):
            selector=lambda name_i: name_i[1]==0
        names=self.names()
        if(shuffle):
            random.shuffle(names)
        train,test=[],[]
        for name_i in names:
            pair_i=(name_i,self[name_i])
            if(selector(name_i)):
                train.append(pair_i)
            else:
                test.append(pair_i)
        return self.__class__(train),self.__class__(test)
    
    def save(self,out_path=None):
        raw_dict={}
        for name_i,data_i in self.items():
            raw_dict[str(name_i)]=list(data_i)
        if(out_path is None):
            return raw_dict
        with open(out_path, 'w') as f:
            json.dump(raw_dict, f)

    def rename(self,name_dict):
        new_feats= self.__class__()
        for name_i,name_j in name_dict.items():
            new_feats[Name(name_j)]=self[name_i]
        return new_feats

    def random(self):
        names=self.names()
        rename_dict={}
        for cat_i,names_i in names.by_cat().items():
            names_i.shuffle()
            for j,name_j in enumerate(names_i):
                new_name_j=f"{cat_i+1}_{j%2}_{len(rename_dict)}"
                rename_dict[name_j]=new_name_j
        return self.rename(rename_dict)

    def concat(self,other_dict):
        concat_data=DataDict()
        for name_i,vec_i in self.items():
            other_i=other_dict[name_i]
            concat_i=np.concatenate([other_i,vec_i])
            concat_data[name_i]=concat_i
        return concat_data    

class DataGroup(list):
    def __new__(cls, name_list=None):
        if(name_list is None):
            name_list=[]
        return list.__new__(cls,name_list)

    def save(self,out_path):
        make_dir(out_path)
        for i,data_i in enumerate(self):
            data_i.convert()
            data_i.save(f'{out_path}/{i}')

def read_data_group(in_path):
    paths=top_files(in_path)
    datasets=[ read_data(path_i) for path_i in paths]
    return DataGroup(datasets)

class NameList(list):
    def __new__(cls, name_list=None):
        if(name_list is None):
            name_list=[]
        return list.__new__(cls,name_list)

    def shuffle(self):
        random.shuffle(self)
        return self

    def n_cats(self):
        return len(self.unique_cats())

    def unique_cats(self):
        return set(self.get_cats())

    def get_cats(self):
        return [name_i.get_cat() for name_i in self]     

    def binarize(self,j):
        return [ int(cat_i==0) for cat_i in self.get_cats()]

    def by_cat(self):
        cat_dict={cat_j:NameList() 
                for cat_j in self.unique_cats()}
        for name_i in self:
            cat_dict[name_i.get_cat()].append(name_i)
        return cat_dict

    def cats_stats(self):
        stats_dict={ cat_i:0 for cat_i in self.unique_cats()}
        for cat_i in self.get_cats():
            stats_dict[cat_i]+=1
        return stats_dict

    def save(self,out_path):
        with open(out_path, 'w') as f:
            json.dump(self, f)

class Name(str):
    def __new__(cls, p_string):
        return str.__new__(cls, p_string)

    def __len__(self):
        return len(self.split('_'))
    
    def get_cat(self):
        return int(self.split('_')[0])-1

    def __getitem__(self,i):
        return int(self.split('_')[i])

    def set_train(self,trainset:bool):
        raw=self.split('_')
        raw[1]=str(int(trainset))
        return Name('_'.join(raw))          

def read_data(in_path):
    with open(in_path, 'r') as f:
        data_dict = json.load(f)
        data_dict={ Name(name_i):np.array(data_i)
            for name_i,data_i in data_dict.items()}
        return DataDict(data_dict)

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    paths=sorted(paths)#,key=natural_keys)
    return paths