data_path='../data'
out='../test'
hyper_path="${out}/hyper"
model_path="${out}/models"
pred_path="${out}/pred"

n_split=10
n_iter=10
dir=1

mkdir ${out}

python3 hyper.py --data "${data_path}" --hyper "${hyper_path}" \
--n_split "${n_split}" --n_iter "${n_iter}" --dir "${dir}"

python3 train.py --data "${data_path}" --hyper "${hyper_path}" \
--models "${model_path}" --n_splits "${n_split}" --n_repeats "${n_iter}" \
--dir "${dir}"

python3 pred.py --data "${data_path}"  --models "${model_path}"  \
--pred "${pred_path}" --dir "${dir}"

python3 eval.py  --pred "${pred_path}" --dir "${dir}"