import argparse
import pred,plot,utils


def metric_plot(conf_dict):
    metric,text=conf_dict["metric"],conf_dict["text"]
    x_clf,y_clf=conf_dict["x_clf"],conf_dict["y_clf"]
    result_dict=pred.unify_results(conf_dict["exp_path"])
    x_dict=result_dict.get_mean_metric(x_clf,metric=metric)
    y_dict=result_dict.get_mean_metric(y_clf,metric=metric)
    if("names" in conf_dict):
        text=conf_dict["names"]
    plot.dict_plot( x_dict,
                    y_dict,
                    x_clf,
                    y_clf,
                    text=text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="sum.json")
    args = parser.parse_args()
    conf_dict=utils.read_json(args.conf)
    metric_plot(conf_dict)