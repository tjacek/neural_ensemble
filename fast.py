import sys
import output,ens

if __name__ == "__main__":
    if(len(sys.argv)>1):
        data_dir= sys.argv[1]
    else:
        data_dir='uci_npz'
    if(len(sys.argv)>2):
        data_dir=[ f'{data_dir}/{path_i}' 
            for path_i in sys.argv[2:]]
    ens_types=[ens.Ensemble,ens.NoEnsemble]
    clf_types=['LR']
    exp=output.ESCFExp(ens_types,clf_types)
    line_dict=exp(data_dir)
    output.format(line_dict)