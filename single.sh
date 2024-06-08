data_path='../uci/old/cleveland'
out_path='../single'
hyper_path="${out_path}/hyper"
model_path="${out_path}/model"
n_split=3
n_iter=3

mkdir ${out_path}

python3 hyper.py --data "${data_path}" --hyper "${hyper_path}"

python3 train.py --data "${data_path}" --hyper "${hyper_path}" --model "${model_path}"

python3 eval.py --data "${data_path}" --model "${model_path}"