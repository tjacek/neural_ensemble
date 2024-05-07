import data,protocol

@utils.DirFun([("data_path",0),("model_path",1)])
def stat_sig(data_path:str,
             model_path:str,
             protocol_obj:protocol.Protocol,
             clf_type="RF"):
    dataset=data.get_data(data_path)
    exp_io= protocol_obj.get_group(exp_path=model_path)
    for nescf_ij in exp_io.iter_necscf(dataset):
    	print(nescf_ij)
if __name__ == '__main__':
