import utils
utils.silence_warnings()
import base,data,exp,deep

def train_data(dataset,protocol,alg_params,hyper_params):
    acc=[]
    for split_i in protocol.iter(dataset):
        model_i=deep.ensemble_builder(params=dataset.params,
        	                          hyper_params=hyper_params,
        	                          alpha=0.5)
        exp_i=exp.Experiment(split=split_i,
        	                  hyper_params=hyper_params,
        	                  model=model_i)
        exp_i.train(alg_params,
        	        verbose=0)
        result_i=exp_i.eval(alg_params)
        result=exp_i.split.eval("RF")
        acc.append(result_i.acc()-result.acc())
    print(acc)

def stat_sig(dataset,protocol,alg_params,hyper_params,clf_type="RF"):
	rf_results,ne_results=[],[]
	for split_i in protocol.iter(dataset):
        model_i=deep.ensemble_builder(params=dataset.params,
        	                          hyper_params=hyper_params,
        	                          alpha=0.5)
        exp_i=exp.Experiment(split=split_i,
        	                  hyper_params=hyper_params,
        	                  model=model_i)
        exp_i.train(alg_params,
        	        verbose=0)
        ne_results.append(  exp_i.eval(alg_params))        
        rf_results.append(exp_i.split.eval(clf_type))

if __name__ == '__main__':
    in_path='../uci/cleveland'
    dataset=data.get_data(in_path)
#    hyper_params={'units_0': 204, 'units_1': 52, 'batch': 0, 'layers': 2}
    hyper_params={'units_0': 123, 'units_1': 65, 'batch': 0, 'layers': 2}

    hyper_dict=train_data(dataset=dataset,
                          protocol=base.Protocol(),
                          alg_params=base.AlgParams(),

                           hyper_params=hyper_params)