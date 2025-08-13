import numpy as np
import argparse
import matplotlib.pyplot as plt
import base,dataset,utils
import pvalue
utils.silence_warnings()

def summary(exp_path):
    result_dict=pvalue.get_result_dict(exp_path)
    def df_helper(clf_type):
        acc_dict=result_dict.get_clf(clf_type,metric="acc")
        balance_dict=result_dict.get_clf(clf_type,metric="balance")
        lines=[]
        for data_i in acc_dict:
            line_i=[data_i,clf_type]
            line_i.append(np.mean(acc_dict[data_i]))
            line_i.append(np.mean(balance_dict[data_i]))
            line_i.append(len(acc_dict[data_i]))
            lines.append(line_i)
        return lines
    df=dataset.make_df(helper=df_helper,
                      iterable=result_dict.clfs(),
                      cols=["data","clf","acc","balance","n_splits"],
                      multi=True)     
    print(df.by_data(sort='acc'))

def plot_box(exp_path):
    result_dict=pvalue.get_result_dict(exp_path)
    fig, ax = plt.subplots()
    data=result_dict.data()
    clf_types=result_dict.clfs()
    step=len(clf_types)
    for i,clf_i in enumerate(clf_types):
        dict_i=result_dict.get_clf(clf_i,metric="acc")
        values_i=[dict_i[data_j] for data_j in data[:5]]
#            print(data_j)
        positions_i=[j*step+i for j,_ in enumerate(data[:5])]

        box_i=ax.boxplot(values_i,
                         positions=positions_i,
                         patch_artist=True)
        plt.setp(box_i['medians'], color="black")
        plt.setp(box_i['boxes'], color='lime')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default="uci_exp/exp")
    args = parser.parse_args()
#    summary(exp_path=args.exp_path)
    plot_box(exp_path=args.exp_path)