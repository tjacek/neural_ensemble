name='lymphography' 
data="uci/${name}"
#name="../../cl/out/${name}"
hyper="${name}/hyper.txt"
models="${name}/models"
log="${name}/log.time"
pvalue="${name}/pvalue.txt"
pred="${name}/pred"
results="${name}/results"
acc_path="${name}/acc.txt"
n_split=10
n_repeats=10
bayes_iter=5

hyper_cmd=true

train_model () {
  python3 train.py --data ${data} --hyper ${hyper} \
    --ens $1 --out "${models}" --log_path ${log} \
    --n_split ${n_split} --n_repeats ${n_repeats} \
    --acc_path ${acc_path}
}

mkdir ${name}

#if [ $hyper_cmd == true ]; then {
#  python3 hyper.py --data ${data} --hyper ${hyper} \
#    --n_split ${n_split} --n_iter ${bayes_iter} \
#    --clfs 'all' --log_path ${log}
#}
#fi

#train_model "GPU,CPU"

#python3 pred.py --data ${data} --models ${models} \
#     --out ${pred} --log_path ${log} 

python3 eval.py --pred ${pred} --results ${results} \
  --p_value ${pvalue}