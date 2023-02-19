import os
import utils,data,ens

class Transform(object):
    def __init__(self,reader=None,writer=None):
        if( reader is None ):
            reader=ens.GzipReader()
        if( writer is None ):
            writer=ens.npz_writer
        self.reader=reader
        self.writer=writer

    def __call__(self,in_path,out_path,depth=3):
        @utils.dir_map(depth)
        def helper(in_path,out_path):
            print(in_path)
            ens_i=self.reader(in_path)
            print(out_path)
            self.writer(ens_i,out_path)
        helper(in_path,out_path)

transform= Transform()
transform('../positional_voting/ECSCF/imb_gzip','test')