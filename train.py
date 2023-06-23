import tools
tools.silence_warnings()
import argparse
import data,deep

def single_exp(data_path,hyper_path):
    X,y=data.get_dataset(data_path)
    hyper_params=parse_hyper(hyper_path)
    dataset_params=data.get_dataset_params(X,y)
    model=deep.simple_nn(dataset_params,hyper_params)
    print(dir(model))

def parse_hyper(hyper_path):
    with open(hyper_path) as f:
        line = eval(f.readlines()[-1])
        hyper_dict=line[0]
        layers= [key_i for key_i in hyper_dict
                   if('unit' in key_i)]
        layers.sort()
        return { 'batch':hyper_dict['batch'],
                 'layers':[hyper_dict[name_j] 
                            for name_j in layers] }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='data/wine-quality-red')
    parser.add_argument("--hyper", type=str, default='hyper.txt')
    args = parser.parse_args()
    single_exp(args.data,args.hyper)