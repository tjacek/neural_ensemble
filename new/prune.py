import tools
tools.silence_warnings()
from time import time
import pred,models

def single_exp(data_path,model_path,out_path):
    X,y=tools.get_dataset(data_path)
    dir_path= '/'.join(model_path.split('/')[:-1])
    acc_dict= pred.read_acc_dict(f'{dir_path}/acc.txt')
    modelsIO=models.ManyClfs(model_path)
    for i,clf_dict_i,train_i,test_i in modelsIO.split(X,y):     
        for name_j,model_j in clf_dict_i.items():            
            print(acc_dict[i])
            binary= model_j.binary_model.predict(test_i[0])
            threshold_i= 1.0 -(1/len(binary))
            s_binary=[binary[k] 
                for k,acc_k in enumerate(acc_dict[i].values())
                    if(acc_k>threshold_i)]
            print(len(s_binary))

if __name__ == '__main__':
    args=pred.parse_args()
    tools.start_log(args.log_path)
    start=time()
    single_exp(args.data,args.models,args.out)
    tools.log_time(f'PRED-SELECT:{args.data}',start) 
