data_path='../../cl/out'
dict="models"

for data_i in "$data_path"/*
do
  dir_i="${data_i}/${dict}"
  rm -r ${dir_i}
done