data="bad_exp/data"
out="bad_exp/exp"
step=10
start=5
clf="TREE-MLP"

python3 train.py --data ${data} --out_path ${out} --start ${start} --clf_type ${clf} --step ${step}
