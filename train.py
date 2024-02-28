import base,data,exp,deep

def train_data(dataset,protocol,alg_params,hyper_params):
    for split_i in protocol.iter(dataset):
        model_i=deep.ensemble_builder(params=dataset.params,
        	                          hyper_params=hyper_params,
        	                          alpha=0.5)
        exp_i=exp.Experiment(split=split_i,
        	                  hyper_params=hyper_params,
        	                  model=model_i)
        exp_i.train(alg_params,
        	        verbose=0)
        exp_i.eval(alg_params)

if __name__ == '__main__':
    in_path='../uci/wall-following'
    dataset=data.get_data(in_path)
    hyper_params={'units_0': 204, 'units_1': 52, 'batch': 0, 'layers': 2}
    hyper_dict=train_data(dataset=dataset,
                          protocol=base.Protocol(),
                          alg_params=base.AlgParams(),

                           hyper_params=hyper_params)