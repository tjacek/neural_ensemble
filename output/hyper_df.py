import pandas as pd
import tools

def hyper_frame(in_path):
    lines=[]
    for path_i in tools.top_files(in_path):
        name_i=path_i.split('/')[-1]
        layers,unit,_=read_params(path_i)
        if(len(unit)==1):
            layers['units_1']=-1
            unit['units_1']=-1
        line=[layers['units_0'],unit['units_0'],
            layers['units_1'],unit['units_1'],layers['batch']]
        lines.append([name_i]+line)
    df = pd.DataFrame(lines,columns=
    	['dataset','units0','layer0','units1','layer1','batch'])
    print(df['layer0'].median())
    print(df[df['layer1']>0]['layer1'].median())


def read_params(hyper_path):
    with open(hyper_path) as f:
        return eval(f.readlines()[-1])

if __name__ == "__main__":
    in_path='hyper'
    hyper_frame(in_path)