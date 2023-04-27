data_path='uci'
out_path='uci_out'
n_split=3
n_repeats=3
bayes_iter=2
clfs='GPUClf_2_2,CPUClf_2'  #'all'
log_path="${out_path}/time.log"

mkdir ${out_path}
mkdir "${out_path}/hyper"
mkdir "${out_path}/models"
mkdir "${out_path}/results"

for data_i in "$data_path"/*
do
  echo "${data_i}"
  IFS="/" read -ra arr <<< "$data_i"
  name_i="${arr[1]}"
  hyper_i="${out_path}/hyper/${name_i}"
  model_i="${out_path}/models/${name_i}"
  results_i="${out_path}/results/${name_i}"

  python3 hyper.py --data "${data_i}" --hyper "${hyper_i}" \
        --n_split ${n_split} --n_iter ${bayes_iter} --clfs ${clfs} \
        --log_path ${log_path}
  python3 train.py --data "${data_i}" --hyper "${hyper_i}" \
        --out "${model_i}" --ens best --n_splits "${n_split}" \
        --n_repeats "${n_repeats}" --log_path ${log_path}
  python3 eval.py --data ${data_i} --models ${model_i} \
        --results ${results_i} --log_path ${log_path}
done