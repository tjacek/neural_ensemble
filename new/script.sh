data_path='../../uci'
out_path='../../acc_min'
n_split=10
n_repeats=10
bayes_iter=20
clfs="all" #'GPU,CPU'  
log_path="${out_path}/time.log"

mkdir ${out_path}

for data_i in "$data_path"/*
do
  echo "${data_i}"
  IFS="/" read -ra arr <<< "$data_i"
  name_i="${arr[-1]}"
  dir_path="${out_path}/${name_i}"
  echo ${dir_path}
  mkdir ${dir_path}

  hyper_i="${out_path}/${name_i}/hyper.txt"
  model_i="${out_path}/${name_i}/models"
  results_i="${out_path}/${name_i}/results"
  pvalue_i="${out_path}/${name_i}/pvalue.txt"
  pred_i="${out_path}/${name_i}/pred"
  acc_i="${out_path}/${name_i}/acc.txt"

#  python3 hyper_binary.py --data "${data_i}" --hyper "${hyper_i}" \
#    --n_split ${n_split} --n_iter ${bayes_iter} --clfs ${clfs} --log_path ${log_path}

#  python3 train.py --data "${data_i}" --hyper "${hyper_i}" \
#        --out "${model_i}" --ens 'CPU' --n_splits "${n_split}" \
#        --n_repeats "${n_repeats}" --log_path ${log_path} --acc_path "${acc_i}"

#  python3 pred.py --data ${data_i} --models ${model_i} \
#     --out ${pred_i} --log_path ${log_path} 

#  python3 eval.py --pred ${pred_i} --results ${results_i} \
#     --p_value ${pvalue_i}

  python3 summary.py --dir ${out_path}  --metric 'acc_mean' --out "${out_path}/summary.txt"
#        --results ${results_i} --p_value ${pvalue_i} --log_path ${log_path}
done