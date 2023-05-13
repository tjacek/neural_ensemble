import tools
tools.silence_warnings()
import argparse
import pred

def single_exp(pred_path,result_path,pvalue_path):
    pred_dict= pred.read_preds(pred_path)
    print(pred_dict)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='vehicle/results2')
    parser.add_argument("--results", type=str, default='vehicle/p_value')
    parser.add_argument("--p_value", type=str, default='vehicle/p_value')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()
    single_exp(args.pred,args.results,args.p_value)