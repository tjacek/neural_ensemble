import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from tensorflow import keras
import json
import data

class EnsFeatures(object):
    def __init__(self,common,binary):
        self.common=common
        self.binary=binary

def eval(data_path,model_path):
    common=data.read_data(data_path)
    ens_feats= gen_feats(common,model_path)
    print(len(ens_feats.binary))

def gen_feats(raw_data,in_path):
    model_path=f'{in_path}/models'
    models=[keras.models.load_model(path_i)
        for path_i in data.top_files(model_path)]  
    rename_path=f'{in_path}/rename'
    with open(rename_path, 'r') as f:
        rename_dict= json.load(f)
    common=raw_data.rename(rename_dict)
    X,y,names=common.as_dataset()
    binary=[]
    for model_i in models:
        X_i=model_i.predict(X)
        print(X_i.shape)
        binary_i=data.from_names(X_i,names)
        binary.append(binary_i)
    return EnsFeatures(common,binary)

data_path='../../uci/json/wine'
model_path='test/0/0'
eval(data_path,model_path)