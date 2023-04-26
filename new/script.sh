data_path='uci'
out_path='uci_out'
n_split=3
n_repeats=3
bayes_iter=5

mkdir ${out_path}
mkdir "${out_path}/hyper"
mkdir "${out_path}/models"

for data_i in "$data_path"/*
do
  echo "${data_i}"
  IFS="/" read -ra arr <<< "$data_i"
  name_i="${arr[1]}"
  hyper_i="${out_path}/hyper/${name_i}"
  model_i="${out_path}/models/${name_i}"
  python3 hyper.py --data "${data_i}" --hyper "${hyper_i}" \
        --n_split ${n_split} --n_iter ${bayes_iter} --clfs all
  python3 train.py --data "${data_i}" --hyper "${hyper_i}" \
        --out "${model_i}" --ens best --n_splits "${n_split}" \
        --n_repeats "${n_repeats}"
done