from sklearn.metrics import accuracy_score

def get_metric(metric_type):
    if(metric_type=='acc'):
	    return accuracy_score