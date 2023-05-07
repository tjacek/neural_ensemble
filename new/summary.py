import argparse
import pandas as pd
import tools

def make_summary(dir_path,out_path,metric='acc_mean'):
    paths=tools.get_dirs(dir_path)
    for path_i in paths:
        name_i=path_i.split('/')[-1]
        result_i=f'{path_i}/results'
        df=pd.read_csv(result_i) 
        df=df.sort_values(by=metric,ascending=False)
        df_pvalue=pd.read_csv(f'{path_i}/pvalue.txt') 
        with open(out_path,"a") as f:
            f.write(f'{name_i}\n')
            f.write(df.to_csv())
            f.write(df_pvalue.to_csv())

def short_summary(dir_path,out_path):
    paths=tools.get_dirs(dir_path)
    cols=['dataset','ens','clf','imprv','diff']
    lines=[]
    for path_i in paths:
        name_i=path_i.split('/')[-1]
        result_i=f'{path_i}/results'
        df=pd.read_csv(result_i) 
        df_pvalue=pd.read_csv(f'{path_i}/pvalue.txt') 
        sig_i=(df_pvalue[df_pvalue['sig']==True])
        for j, row_j in sig_i.iterrows():
            ens_j,clf_j=row_j['ens'],row_j['clf']
#            if((clf_j in ens_j) ):#or ('TF' in ens_j)):

            ens_acc= df[df['clf']==ens_j]['acc_mean']
            clf_acc= df[df['clf']==clf_j]['acc_mean']
            diff= (float(ens_acc)-float(clf_acc))
            line_j=[name_i,ens_j,clf_j,(diff>0),diff ]
            lines.append(line_j)
    df= pd.DataFrame(lines,columns=cols)
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default='../../cl/out')
    parser.add_argument("--metric", type=str, default='balanced_acc_mean')

    parser.add_argument("--out", type=str, default='../../cl/out/summary.txt')
    parser.add_argument("--short",action='store_true')

    args = parser.parse_args()
    if(args.short):
        short_summary(args.dir,args.out)
    else:
        make_summary(args.dir,args.out,args.metric)