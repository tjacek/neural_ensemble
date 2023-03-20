#!/bin/bash
conf_path=conf/l1.cfg
n_iters=3
n_split=3

echo 'conf path' ${conf_path}
echo 'n_iters' ${n_iters}
echo 'n_split' ${n_split}

exp(){
  if [ $2 != 'default' ]; then 
  { 
  	echo 'Optimisation of hyperparametrs';
    python hyper.py --conf ${conf_path} --n_split ${n_split} \
        --dir_path $1   --optim_type $2;
    echo 'Training models';
    python train.py --conf ${conf_path} --n_iters ${n_iters} \
        --n_split ${n_split} --dir_path $1  
  } 
#  fi

  elif [ $2 == 'default' ]; then 
  { 
    echo 'Training models';
    python train.py --default --conf ${conf_path} \
     --n_iters ${n_iters}  --n_split ${n_split} --dir_path $1  
  }
  fi
  echo 'Test model';
  python test.py --conf ${conf_path} --dir_path $1
  echo 'Genreate plot';
  python output/plot.py --conf ${conf_path} --dir_path $1
  echo 'Genreate confusion matrix';
  python output/cf.py --conf ${conf_path} --dir_path $1	
}

dir='../small'

exp "${dir}/default" 'default'
exp "${dir}/grid" 'grid'
exp "${dir}/bayes" 'bayes'