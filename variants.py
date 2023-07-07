from collections import defaultdict
import data,pred,tools

def single_exp(data_path,model_path,out_path):
    X,y=data.get_dataset(data_path)
    pred_dict=defaultdict(lambda:[])
    for name_i,model_i,split_i in pred.get_model_paths(model_path):
        if('binary' in name_i):
            train,test=split_i.get_dataset(X,y)
            cs_train=model_i.extract(train.X)
            cs_test=model_i.extract(test.X)
            pred_k=pred.necscf(train,test,cs_train,cs_test,"LR")
            pred_dict[f'{name_i}(LR)'].append(pred_k)
    tools.make_dir(out_path)
    for name_i,pred_i in pred_dict.items():
        pred.save_pred(f'{out_path}/{name_i}',pred_i)

if __name__ == '__main__':
#    if(args.dir>0):
    single_exp=tools.dir_fun(3)(single_exp)
    single_exp('../uci',"../10_10/models","pred")