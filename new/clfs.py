import inspect
import ens

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

def params_names(clf_i):
    sig_i=inspect.signature(clf_i.__init__)
    return list(sig_i.parameters.keys())

#def get_desc()

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