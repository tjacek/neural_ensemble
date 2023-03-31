#!/bin/bash
conf_path=conf/ovo.cfg
dir='../small'
n_iters=3
n_split=3
clf_jobs=1
hyper_jobs=5
batch_ratio=0.5

echo 'conf path' ${conf_path}
echo 'n_iters' ${n_iters}
echo 'n_split' ${n_split}
echo 'clf_jobs' ${clf_jobs}
echo 'hyper_jobs' ${hyper_jobs}
echo 'batch_ratio' ${batch_ratio}

exp(){
  start_time="$(date -u +%s)"
  if [ $2 != 'default' ]; then 
  { 
  	echo 'Optimisation of hyperparametrs';
    python hyper_keras.py --conf ${conf_path} --dir_path $1
#    --conf ${conf_path} --n_split ${n_split} \
#        --dir_path $1   --optim_type $2 --clf_jobs ${clf_jobs} \
#        --hyper_jobs ${hyper_jobs}  --batch_ratio ${batch_ratio};

#    python hyper.py --conf ${conf_path} --n_split ${n_split} \
#        --dir_path $1   --optim_type $2 --clf_jobs ${clf_jobs} \
#        --hyper_jobs ${hyper_jobs}  --batch_ratio ${batch_ratio};

    echo 'Training models';
    python train.py --conf ${conf_path} --n_iters ${n_iters} \
       --lazy --n_split ${n_split} --dir_path $1 \
       --clf_jobs ${clf_jobs} --batch_ratio ${batch_ratio};
  } 
  elif [ $2 == 'default' ]; then 
  { 
    echo 'Training models';
    python train.py --default --conf ${conf_path} --lazy \
     --n_iters ${n_iters}  --n_split ${n_split} --dir_path $1 \
     --clf_jobs ${clf_jobs} --batch_ratio ${batch_ratio}; 
  }
  fi
  eval_model $1
  
  end_time="$(date -u +%s)"
  elapsed="$(($end_time-$start_time))"  
  echo "Elapsed" ${elapsed}
}

eval_model(){
  echo 'Test model';
  python test.py --conf ${conf_path} --dir_path $1 --clf_jobs ${clf_jobs} \
          --batch_ratio ${batch_ratio};
  echo 'Genreate plot';
  python output/plot.py --conf ${conf_path} --dir_path $1
  echo 'Genreate confusion matrix';
  python output/cf.py --conf ${conf_path} --dir_path $1 
}

elapsed=0
#exp "${dir}/ovo" 'default'
#elapsed1=${elapsed}
exp "${dir}/hyper" 'hyper'
elapsed2=${elapsed}
#exp "${dir}/bayes" 'bayes'
#elapsed3=${elapsed}

#echo "Time default ${elapsed1}"
echo "Time grid ${elapsed2}"
#echo "Time bayes ${elapsed3}"

#eval_model "${dir}/default" 
#eval_model "${dir}/grid"
#eval_model "${dir}/bayes" #'bayes'