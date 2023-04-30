name='newthyroid'
data="uci/${name}"
hyper="${name}/hyper.txt"
models="${name}/models"
log="${name}/log.time"
results="${name}/results"
n_split=10
n_repeats=10
bayes_iter=20

mkdir ${name}
#python3 hyper.py --data ${data} --hyper ${hyper} \
#    --n_split ${n_split} --n_iter ${bayes_iter} \
#    --clfs 'all' --log_path ${log}
#python3 train.py --data ${data} --hyper ${hyper} \
#    --ens 'best' --out ${models} --log_path ${log} \
#    --n_split ${n_split} --n_repeats ${n_repeats} \
python3 eval.py --data ${data} --models ${models} \
     --results ${results} --log_path  ${log}