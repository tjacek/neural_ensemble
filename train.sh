conf_path='conf/ova.cfg'
data_dir='../uci/json'
main_dir='../uci/keras'
n_iters=10
n_splits=10
batch_size=320
datasets='cleveland,wine' # set to '-' to train on all datasets

echo 'conf path' ${conf_path}
echo 'data_dir' ${data_dir}
echo 'main_dir' ${main_dir}
echo 'n_iters' ${n_iters}
echo 'n_splits' ${n_splits}
echo 'batch_size' ${batch_size}
echo 'datasets' ${datasets}

echo 'Training models';
start_time="$(date -u +%s)"
python3 train.py --conf ${conf_path} --data_dir $data_dir \
--main_dir $main_dir --batch_size $batch_size \
--n_iters ${n_iters} --n_splits $n_splits --datasets $datasets

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"  
echo "Elapsed" ${elapsed}