import inspect,json
import ens,tools

CLFS_NAMES=['GPUClf_2_2','GPUClf_2_1','GPUClf_1_2','GPUClf_1_1',
            'CPUClf_2','CPUClf_1']

class GPUClf_2_2(ens.NeuralEnsembleGPU):
    def __init__(self,binary1=1,binary2=1,multi1=1,multi2=1):
        self.binary1=binary1
        self.binary2=binary2
        self.multi1=multi1
        self.multi2=multi2
        binary=ens.BinaryBuilder([binary1,binary2])
        multi=ens.MultiInputBuilder([multi1,multi2])
        super(GPUClf_2_2, self).__init__(binary,multi)

class GPUClf_2_1(ens.NeuralEnsembleGPU):
    def __init__(self,binary1=1,binary2=1,multi1=1):
        self.binary1=binary1
        self.binary2=binary2
        self.multi1=multi1
        binary=ens.BinaryBuilder([binary1,binary2])
        multi=ens.MultiInputBuilder([multi1])
        super(GPUClf_2_1, self).__init__(binary,multi)

class GPUClf_1_2(ens.NeuralEnsembleGPU):
    def __init__(self,binary1=1,multi1=1,multi2=1):
        self.binary1=binary1
        self.multi1=multi1
        self.multi2=multi2
        binary=ens.BinaryBuilder([binary1])
        multi=ens.MultiInputBuilder([multi1,multi2])
        super(GPUClf_1_2, self).__init__(binary,multi)

class GPUClf_1_1(ens.NeuralEnsembleGPU):
    def __init__(self,binary1=1,multi1=1):
        self.binary1=binary1
        self.multi1=multi1
        binary=ens.BinaryBuilder([binary1])
        multi=ens.MultiInputBuilder([multi1])
        super(GPUClf_1_1, self).__init__(binary,multi)

class CPUClf_2(ens.NeuralEnsembleCPU):
    def __init__(self,binary1=1,binary2=1,multi_clf='RF'):
        self.binary1=binary1
        self.binary2=binary2
        self.multi_clf=multi_clf
        binary=ens.BinaryBuilder([binary1,binary2])
        super(CPUClf_2, self).__init__(binary,multi_clf)

class CPUClf_1(ens.NeuralEnsembleCPU):
    def __init__(self,binary1=1,multi_clf='RF'):
        self.binary1=binary1
        self.multi_clf=multi_clf
        binary=ens.BinaryBuilder([binary1])
        super(CPUClf_1, self).__init__(binary,multi_clf)

def is_cpu(clf_i):
    ens_name=clf_i.__class__.__name__
    return ('CPU' in ens_name)

def read_clf(in_path):
    with open(f'{in_path}/desc', 'r') as f:        
        json_bytes = f.read()                      
        json_str = json_bytes#.decode('utf-8')           
        desc = json.loads(json_str)
        clf_i=get_ens(desc['name'],desc['hyper'])
        clf_i.data_params={}
        for key_i,value_i in desc['data'].items():
            if(type(value_i)==str and value_i.isdigit()):#!='class_weights'):
                value_i=int(value_i)
            clf_i.data_params[key_i]=value_i
        clf_i.data_params['class_weights']={
             int(key_i):value_i
           for key_i,value_i in clf_i.data_params['class_weights'].items()}
        clf_i.empty_model()
        clf_i.load_weights(in_path)
        print(str(clf_i))
        return clf_i

def save_clf(clf_i,out_path):
    clf_i.save_weights(out_path) 
    desc=get_desc(clf_i)
    with open(f'{out_path}/desc', 'wb') as f:
        json_str = json.dumps(desc, default=str)         
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)

def params_names(clf_i):
    sig_i=inspect.signature(clf_i.__init__)
    return list(sig_i.parameters.keys())

def get_desc(clf_i):
    clf_name=str(clf_i.__class__.__name__)
    hyper={ name_i:getattr(clf_i, name_i) 
        for name_i in params_names(clf_i)}
    
    return {'name':clf_name,'hyper':hyper,
        'data':clf_i.data_params}

def get_ens(name_i,hyper=None):
    if(name_i=='GPUClf_2_2'):
        ens_i=GPUClf_2_2
    if(name_i=='GPUClf_2_1'):
    	ens_i=GPUClf_2_1
    if(name_i=='GPUClf_1_2'):
    	ens_i=GPUClf_1_2
    if(name_i=='GPUClf_1_1'):
    	ens_i=GPUClf_1_1
    if(name_i=='CPUClf_2'):
        ens_i=CPUClf_2
    if(name_i=='CPUClf_1'):
        ens_i=CPUClf_1
    if(hyper is None):
        return ens_i()
    return ens_i(**hyper)	