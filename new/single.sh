name='wine-quality-red' 
data="uci/${name}"
hyper="${name}/hyper.txt"
models="${name}/models"
log="${name}/log.time"
pvalue="${name}/pvalue.txt"
results="${name}/results"
n_split=3
n_repeats=3
bayes_iter=5

hyper_cmd=false

train_model () {
python3 train.py --data ${data} --hyper ${hyper} \
    --ens $1 --out "${models}_$1" --log_path ${log} \
    --n_split ${n_split} --n_repeats ${n_repeats} 
}


mkdir ${name}

#if [ $hyper_cmd == true ]; then {
#  python3 hyper_keras.py --data ${data} --hyper ${hyper} \
#    --n_split ${n_split} --n_iter ${bayes_iter} \
#    --clfs 'all' --log_path ${log}
#}

train_model "GPU"
train_model "CPU"

#python3 eval.py --data ${data} --models ${models} \
#     --results ${results} --log_path  ${log} --p_value ${pvalue}