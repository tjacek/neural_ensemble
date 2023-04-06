conf_path='conf/ovo.cfg'
data_dir='../uci/json'
main_dir='../uci/keras'
n_splits=10
batch_size=320

echo 'conf path' ${conf_path}
echo 'data_dir' ${data_dir}
echo 'main_dir' ${main_dir}
echo 'n_splits' ${n_splits}
echo 'batch_size' ${batch_size}

echo 'Optimisation of hyperparametrs';
start_time="$(date -u +%s)"
python3 hyper_keras.py --conf ${conf_path} --data_dir $data_dir \
--main_dir $main_dir --batch_size $batch_size --n_splits $n_splits

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"  
echo "Elapsed" ${elapsed}