import os
import utils,data,ens

class Transform(object):
    def __init__(self,reader=None,writer=None):
        if( reader is None ):
            reader=ens.GzipReader()
        if( writer is None ):
            writer=ens.npy_writer
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
#    factory=ens.EnsembleFactory()
#    paths=data.top_files(f'{in_path}/feats')
#    data.make_dir(out_path)
#    for i,path_i in enumerate(paths):
#        out_i=f'{out_path}/{i}'
#        ens_i=factory(path_i)
#        ens_i.as_gzip(out_i)
#        print(out_i)

transform= Transform()
transform('../positional_voting/ECSCF/imb_gzip','test')