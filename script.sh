#!/bin/bash
conf_path=conf/base.cfg
n_iters=3
n_split=3

echo 'conf path' ${conf_path}
echo 'n_iters' ${n_iters}
echo 'n_split' ${n_split}

echo 'Optimisation of hyperparametrs'
python hyper.py --conf ${conf_path} --n_split ${n_split}
echo 'Training models'
python train.py --conf ${conf_path} --n_iters ${n_iters}  --n_split ${n_split}
echo 'Test model'
python test.py --conf ${conf_path}
echo 'Genreate plot'
python output/plot.py --conf ${conf_path}
echo 'Genreate confusion matrix'
python output/cf.py --conf ${conf_path}