data="bad_exp/data"
out="bad_exp/exp"
step=10

clfs=("TREE-MLP" "TREE-ENS" "BINARY-TREE-ENS")

for clf_i in "${clfs[@]}"
do
    for j in {0..3}
    do
        start=$((step * j))
        python3 train.py --data ${data} --out_path ${out} --start ${start} --clf_type ${clf_i} --step ${step}
    done
done