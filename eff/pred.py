import tools
tools.silence_warnings()
import os,argparse

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
    dataset=data.get_dataset(data_path)
    accuracy=tools.get_metric('acc')
    for model_path_i in tools.top_files(model_path):
        deep_ens=ens.read_ens(model_path_i)
        cs_feats_i=deep_ens.extract(dataset.X)
        learn.make_features(dataset=dataset,
                            split=deep_ens.split,
                            cs_feats=cs_feats_i)
#        necscf=learn.NECSCF(dataset=dataset,
#                            split=deep_ens.split,
#                            cs_feats=cs_feats_i)
#        necscf('LR')


if __name__ == '__main__':
    dir_path='../../optim_alpha/s_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../s_uci')
    parser.add_argument("--models", type=str, default=f'models')
    parser.add_argument("--pred", type=str, default=f'pred')
    parser.add_argument("--log", type=str, default=f'log.info')
    args = parser.parse_args()
#    tools.start_log(args.log)
    extract_exp(data_path=args.data,
                model_path=args.models,
                pred_path=args.pred)