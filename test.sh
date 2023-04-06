conf_path='conf/ova.cfg'
data_dir='../uci/json'
main_dir='../uci/keras'
batch_size=320
gen_output=false

echo 'conf path' ${conf_path}
echo 'data_dir' ${data_dir}
echo 'main_dir' ${main_dir}
echo 'batch_size' ${batch_size}

echo 'Testing models';
start_time="$(date -u +%s)"
if [ $gen_output == true ]; then {
  python3 test.py --conf ${conf_path} --data_dir $data_dir \
  --main_dir $main_dir --batch_size $batch_size; 
}
fi

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"  
echo "Elapsed" ${elapsed}
python3 results.py --conf ${conf_path} --main_dir $main_dir