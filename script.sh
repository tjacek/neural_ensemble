#!/bin/bash
conf_path=conf/small.cfg
dir='../small'
n_iters=3
n_split=3

echo 'conf path' ${conf_path}
echo 'n_iters' ${n_iters}
echo 'n_split' ${n_split}

exp(){
  start_time="$(date -u +%s)"
  if [ $2 != 'default' ]; then 
  { 
  	echo 'Optimisation of hyperparametrs';
    python hyper.py --conf ${conf_path} --n_split ${n_split} \
        --dir_path $1   --optim_type $2;
    echo 'Training models';
    python train.py --conf ${conf_path} --n_iters ${n_iters} \
       --lazy --n_split ${n_split} --dir_path $1  
  } 
  elif [ $2 == 'default' ]; then 
  { 
    echo 'Training models';
    python train.py --default --conf ${conf_path} --lazy \
     --n_iters ${n_iters}  --n_split ${n_split} --dir_path $1  
  }
  fi
  eval_model $1
  
  end_time="$(date -u +%s)"
  elapsed="$(($end_time-$start_time))"  
  echo "Time ${2} ${elapsed}"
}

eval_model(){
  echo 'Test model';
  python test.py --conf ${conf_path} --dir_path $1
  echo 'Genreate plot';
  python output/plot.py --conf ${conf_path} --dir_path $1
  echo 'Genreate confusion matrix';
  python output/cf.py --conf ${conf_path} --dir_path $1 
}

exp "${dir}/default" 'default'
exp "${dir}/grid" 'grid'
exp "${dir}/bayes" 'bayes'

#eval_model "${dir}/default" 
#eval_model "${dir}/grid"
#eval_model "${dir}/bayes" #'bayes'