data_path='uci'
out_path='../../cl/out'
n_split=3
n_repeats=3
bayes_iter=2
clfs='GPU,CPU'  #'all'
log_path="${out_path}/time.log"

for data_i in "$data_path"/*
do
  echo "${data_i}"
  IFS="/" read -ra arr <<< "$data_i"
  name_i="${arr[1]}"
  hyper_i="${out_path}/${name_i}/hyper.txt"
  model_i="${out_path}/${name_i}/models"
  results_i="${out_path}/${name_i}/results"
  pvalue_i="${out_path}/${name_i}/pvalue.txt"

#  python3 hyper.py --data "${data_i}" --hyper "${hyper_i}" \
#        --n_split ${n_split} --n_iter ${bayes_iter} --clfs ${clfs} \
#        --log_path ${log_path}
  python3 train.py --data "${data_i}" --hyper "${hyper_i}" \
        --out "${model_i}" --ens 'GPU,CPU' --n_splits "${n_split}" \
        --n_repeats "${n_repeats}" --log_path ${log_path}
  python3 eval.py --data ${data_i} --models ${model_i} \
        --results ${results_i} --p_value ${pvalue_i} --log_path ${log_path}
done