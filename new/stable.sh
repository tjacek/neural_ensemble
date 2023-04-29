data='uci/wine-quality-red'
hyper_file='hyper_stable.txt'
hyper_iters=10
n_split=10
n_iter=20

for (( i=1; i<=hyper_iters; i++ ))
do  
  python3 hyper_keras.py --data "${data}" --hyper "${hyper_file}" \
  --n_iter "${n_iter}" --n_split "${n_split}"
done