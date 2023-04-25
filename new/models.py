import numpy as np
import clfs,tools

class ModelIO(object):
    def __init__(self,dir_path):
        tools.make_dir(dir_path)
        self.dir_path=dir_path

    def save(self,clf_i,i,split_i):
        out_i=f'{self.dir_path}/{i}'
        clfs.save_clf(clf_i,out_i)
        np.save(f'{out_i}/train',split_i[0])
        np.save(f'{out_i}/test',split_i[1])