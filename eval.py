import numpy as np
import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default="binary_exp/exp")
    args = parser.parse_args()
    summary(exp_path=args.exp_path)