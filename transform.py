import os
import utils,data,ens

class Transform(object):
    def __init__(self,reader=None,writer=None):
        if( reader is None ):
            reader=ens.gzip_reader #GzipReader()
        if( writer is None ):
            writer=ens.npz_writer
        self.reader=reader
        self.writer=writer

    def __call__(self,in_path,out_path,depth=3):
        @utils.dir_map(depth)
        def helper(in_path,out_path):
            print(in_path)
            common,binary=self.reader(in_path)
            ens_i=ens.Ensemble(common,binary)
            print(out_path)
            self.writer(ens_i,out_path)
        helper(in_path,out_path)

transform= Transform()
transform('../uci_gzip','../uci_npz')#'../positional_voting/ECSCF/imb_gzip','test')