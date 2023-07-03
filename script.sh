data_path='../data/cleveland'
out='../single'
hyper_path="${out}/hyper"
model_path="${out}/models"
pred_path="${out}/pred"
log_path="${out}/log.info"

n_split=3
n_iter=3
dir=0

mkdir ${out}

#python3 hyper.py --data "${data_path}" --hyper "${hyper_path}" \
#--n_split "${n_split}" --n_iter "${n_iter}" --dir "${dir}" \
#--log "${log_path}"

#python3 train.py --data "${data_path}" --hyper "${hyper_path}" \
#--models "${model_path}" --n_splits "${n_split}" --n_repeats "${n_iter}" \
#--dir "${dir}" --log "${log_path}"

#python3 pred.py --data "${data_path}"  --models "${model_path}"  \
#--pred "${pred_path}" --dir "${dir}" --log "${log_path}"

python3 eval.py  --pred "${pred_path}" --dir "${dir}"