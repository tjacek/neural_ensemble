data_path='../uci/old/'
out_path='../multiple'
hyper_path="${out_path}/hyper"
model_path="${out_path}/model"
n_split=3
n_iter=3

mkdir ${out_path}

python3 hyper.py --data "${data_path}" --hyper "${hyper_path}" --n_split "${n_split}" --n_iter "${n_iter}" --multi

#python3 train.py --data "${data_path}" --hyper "${hyper_path}" --model "${model_path}" --n_split "${n_split}" --n_iter "${n_iter}" --multi

#python3 eval.py --data "${data_path}" --model "${model_path}" --n_split "${n_split}" --n_iter "${n_iter}" --multi