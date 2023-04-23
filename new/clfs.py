import ens

class GPUClf_2_2(ens.NeuralEnsembleGPU):
    def __init__(self,binary1=1,binary2=1,multi1=1,multi2=1):
        self.binary1=binary1
        self.binary2=binary2
        self.multi1=multi1
        self.multi2=multi2
        binary=ens.BinaryBuilder([binary1,binary2])
        multi=ens.MultiInputBuilder([multi1,multi2])
        super(GPUClf_2_2, self).__init__(binary,multi)
    
    def params_names(self):
        return ['binary1','binary2','multi1','multi2']

    def is_cpu(self):
    	return False

class CPUClf_2(ens.NeuralEnsembleCPU):
    def __init__(self,binary1=1,binary2=1,multi_clf='RF'):
        self.binary1=binary1
        self.binary2=binary2
        self.multi_clf=multi_clf
        binary=ens.BinaryBuilder([binary1,binary2])
        super(CPUClf_2, self).__init__(binary,multi_clf)

    def params_names(self):
        return ['binary1','binary2','multi_clf']

    def is_cpu(self):
    	return True

def get_ens(name_i,hyper=None):
    if(name_i=='GPUClf_2_2'):
        ens_i=GPUClf_2_2
    if(name_i=='CPUClf_2'):
        ens_i=CPUClf_2
    if(hyper is None):
        return ens_i()
    return ens_i(**hyper)	