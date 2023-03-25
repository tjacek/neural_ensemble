#!/bin/bash
conf_path=conf/small.cfg
dir='../small'
n_iters=3
n_split=3
clf_jobs=1
hyper_jobs=3

echo 'conf path' ${conf_path}
echo 'n_iters' ${n_iters}
echo 'n_split' ${n_split}
echo 'clf_jobs' ${clf_jobs}
echo 'hyper_jobs' ${hyper_jobs}

exp(){
  start_time="$(date -u +%s)"
  if [ $2 != 'default' ]; then 
  { 
  	echo 'Optimisation of hyperparametrs';
    python hyper.py --conf ${conf_path} --n_split ${n_split} \
        --dir_path $1   --optim_type $2 --clf_jobs ${clf_jobs} \
        --hyper_jobs ${hyper_jobs};
    echo 'Training models';
    python train.py --conf ${conf_path} --n_iters ${n_iters} \
       --lazy --n_split ${n_split} --dir_path $1 --clf_jobs ${clf_jobs} 
  } 
  elif [ $2 == 'default' ]; then 
  { 
    echo 'Training models';
    python train.py --default --conf ${conf_path} --lazy \
     --n_iters ${n_iters}  --n_split ${n_split} --dir_path $1 \
     --clf_jobs ${clf_jobs}  
  }
  fi
  eval_model $1
  
  end_time="$(date -u +%s)"
  elapsed="$(($end_time-$start_time))"  
  echo "Time ${2} ${elapsed}"
}

eval_model(){
  echo 'Test model';
  python test.py --conf ${conf_path} --dir_path $1 --clf_jobs ${clf_jobs}
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