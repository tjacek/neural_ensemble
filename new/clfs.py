import ens

class LargeGPUClf(ens.NeuralEnsembleGPU):
    def __init__(self,binary1=1,binary2=1,multi1=1,multi2=1):
        self.binary1=binary1
        self.binary2=binary2
        self.multi1=multi1
        self.multi2=multi2
        binary=ens.BinaryBuilder([binary1,binary2])
        multi=ens.MultiInputBuilder([multi1,multi2])
        super(LargeGPUClf, self).__init__(binary,multi)

    def params_names(self):
        return ['binary1','binary2','multi1','multi2']	