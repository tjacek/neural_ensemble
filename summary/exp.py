import utils

SEP='|'

class ExpID(str):
    def __getitem__(self, key):
        return self.split(SEP)[key]

def read_acc(pred_path,metric_type='acc'):
    metric_i=utils.get_metric(metric_type)
    for path_i in utils.top_files(pred_path):
        data_i=path_i.split('/')[-1]
        for path_j in utils.top_files(path_i):
            exp_id=[data_i]+path_j.split('/')[-1].split('-')
            exp_id=ExpID('|'.join(exp_id)) 	
            all_pred=utils.read_pred(path_j)
            print(exp_id)
            
dir_path='../../pred'
read_acc(dir_path)