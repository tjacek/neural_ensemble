import argparse
import pandas as pd
import numpy as np
import tools,pred

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

            ens_acc= df[df['clf']==ens_j]['balanced_acc_mean']#['acc_mean']
            clf_acc= df[df['clf']==clf_j]['balanced_acc_mean']#['acc_mean']
            diff= (float(ens_acc)-float(clf_acc))
            line_j=[name_i,ens_j,clf_j,(diff>0),diff ]
            lines.append(line_j)
    df= pd.DataFrame(lines,columns=cols)
    print(df)

def best(dir_path,metric='balanced_acc_mean'):
    paths=tools.get_dirs(dir_path)
    cols=['dataset','best','second','p_value','sig']
    lines=[]
    for path_i in paths:
        name_i=path_i.split('/')[-1]
        result_i=f'{path_i}/results'
        df=pd.read_csv(result_i) 
        df['ens']=df['clf'].apply(lambda clf_i: ('NECSCF' in clf_i))
        df_pvalue=pd.read_csv(f'{path_i}/pvalue.txt') 
        df=df.sort_values(by=metric,ascending=False)

        best,is_ens=df.iloc[0]['clf'],df.iloc[0]['ens']
        
        tmp= df[df['ens']==(not is_ens)].sort_values(by=metric,ascending=False)
        second= tmp.iloc[0]['clf']

        ens_type,clf_type = (best,second)  if(is_ens) else (second,best)

        p_row=df_pvalue[ (df_pvalue['ens']==ens_type) &
                    (df_pvalue['clf']==clf_type)]
#        df_pvalue[]
        lines.append([name_i,best,second,float(p_row['p_value']),str(p_row['sig'])])
    df= pd.DataFrame(lines,columns=cols)
    print(df)

def acc_summary(dir_path):
    for path_i in tools.get_dirs(dir_path):
        print(path_i)
        acc_path_i=f'{path_i}/acc.txt'
        acc_dict=pred.read_acc_dict(acc_path_i)
        for j,clf_j in acc_dict.items():
            acc_j= list(clf_j.values())
            stats_j=[f'{fun(acc_j):.2f}' for fun in [np.mean,np.median,np.max,np.min]]
            print(','.join(stats_j))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default='../../out')
    parser.add_argument("--metric", type=str, default='balanced_acc_mean')

    parser.add_argument("--out", type=str, default='../../out/summary.txt')
    parser.add_argument("--short",action='store_true')
    parser.add_argument("--acc",action='store_true')

    args = parser.parse_args()
    if(args.short):
        short_summary(args.dir,args.out)
    elif(args.acc):
        acc_summary(args.dir)
    else:
        make_summary(args.dir,args.out,args.metric)
#    best(args.dir)