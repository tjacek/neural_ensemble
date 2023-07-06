import numpy as np
from scipy import stats
import tools

def sig_group(in_path):
    acc_dict= tools.metric_dict('acc',in_path)
    mean_dict={ name_i:np.mean(acc_i) 
        for name_i,acc_i in acc_dict.items()}
    def helper(x,y):
        r=stats.ttest_ind(acc_dict[x], 
                acc_dict[y], equal_var=False)
        return r[1]#round(r[1],4)
    while(len(mean_dict)>0):
        group,worse=select_sig(mean_dict,helper)
        mean_dict={name_i:mean_dict[name_i] 
            for name_i in worse}
        print(group)
#        print(worse)

def select_sig(mean_dict,helper):
    names,score = zip(*mean_dict.items())
    k=np.argmax(score)
    best=names[k]
    group,worse=[best],[]
    for name_i in names:
        if(name_i==best):
        	pass
        elif(helper(best,name_i)<0.05 ):
        	group.append(name_i)
        else:
        	worse.append(name_i)
    return group,worse



if __name__ == "__main__":
    in_path='pred'
    for path_i in tools.top_files(in_path):
        sig_group(path_i)
#    print(acc_dict)