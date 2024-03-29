import tools
tools.silence_warnings()
import os,argparse,json
import data,ens,learn,tools

def pred_exp(data_path,hyper_path,model_path):
    dataset=data.get_dataset(data_path)
    accuracy=tools.get_metric('acc')
    for model_path_i in tools.top_files(model_path):
        deep_ens=ens.read_ens(model_path_i)
        y_pred=deep_ens.predict_classes(dataset.X)
        acc_i=accuracy(dataset.y,y_pred)
        print(acc_i) 
        deep_ens.extract(dataset.X)

def extract_exp(data_path,model_path,pred_path):
    acc=tools.get_metric('acc')
    clfs=['LR','SVC']
    variants={'common':learn.common_variant,
              'necscf':learn.necscf_variant,
              'cs':learn.cs_variant}
    factory= learn.FeaturesFactory()
    tools.make_dir(pred_path)
    for model_j in tools.top_files(model_path):
        if(not 'weighted' in model_j):
            continue
        ens_j=model_j.split('/')[-1]
        @tools.log_time(task=f'PRED-{ens_j}')
        def helper(data_path,model_path,pred_path):
            dataset=data.get_dataset(data_path)
            tools.make_dir(pred_path)
            for i,model_path_i in enumerate(tools.top_files(model_path)):
                deep_ens=ens.read_ens(model_path_i)
                cs_feats_i=deep_ens.extract(dataset.X)
                feats=factory(dataset=dataset,
                          split=deep_ens.split,
                          cs_feats=cs_feats_i)
                tools.make_dir(f'{pred_path}/{i}')
                for variant_j,clf_j,pred_j in feats(clfs,variants):
                    id_j=f'{variant_j}-{clf_j}'
                    print(id_j)
                    save_pred(f'{pred_path}/{i}/{id_j}',pred_j) 
        if(os.path.isdir(data_path)):
            helper=tools.dir_fun(3)(helper)
        helper(data_path,model_j,f'{pred_path}/{ens_j}')

def save_pred(out_path,pred_i):        
    with open(out_path, 'wb') as f:
        all_pred=[]
        for test_i,pred_i in pred_i:
            if(type(test_i)!=list):
                test_i=test_i.tolist()
            if(type(pred_i)!=list):
                pred_i=pred_i.tolist()
            all_pred.append((test_i,pred_i))                
        json_str = json.dumps(all_pred, default=str)         
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)

def read_pred(path_i):
    with open(path_i, 'r') as f:        
        json_bytes = f.read()                      
        return json.loads(json_bytes)

if __name__ == '__main__':
    dir_path='../../s_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../s_uci')
    parser.add_argument("--models", type=str, default=f'{dir_path}/models')
    parser.add_argument("--pred", type=str, default=f'{dir_path}/pred')
    parser.add_argument("--log", type=str, default=f'{dir_path}/log.info')
    args = parser.parse_args()
    tools.start_log(args.log)
    extract_exp(data_path=args.data,
                model_path=args.models,
                pred_path=args.pred)