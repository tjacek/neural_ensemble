data_path='../uci/old/cleveland'
out_path='../single'
hyper_path="${out_path}/hyper"
n_split=3
n_iter=3

mkdir ${out_path}

python3 hyper.py --data "${data_path}" --hyper "${hyper_path}"