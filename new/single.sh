name="vehicle"
data="uci/${name}"
#name="../../cl/out/${name}"
hyper="${name}/hyper.txt"
models="${name}/models"
log="${name}/log.time"
acc_path="${name}/acc.txt"
result_dir="best"
n_split=10
n_repeats=10
bayes_iter=20
hyper_cmd=true

mkdir ${name}

#python3 hyper_binary.py --data ${data} --hyper ${hyper} \
#   --n_split ${n_split} --n_iter ${bayes_iter} \
#   --clfs 'all' --log_path ${log}

python3 train.py --data ${data} --hyper ${hyper} \
    --ens 'CPU' --out "${models}" --log_path ${log} \
    --n_split ${n_split} --n_repeats ${n_repeats} \
    --acc_path ${acc_path}

pred_dir="${name}/${result_dir}"
mkdir "${pred_dir}"
p_value="${pred_dir}/pvalue.txt"
results="${pred_dir}/results"
pred="${pred_dir}/pred"

python3 prune.py --data ${data} --models ${models} \
    --out ${pred} --log_path ${log}
python3 eval.py --pred ${pred} --results ${results} \
  --p_value ${p_value}