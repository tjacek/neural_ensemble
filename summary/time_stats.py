import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))
import re
import utils

def time_stats(in_path,info='TRAIN'):
    pattern=re.compile(r"(\d*\.)?\d+")
    @utils.dir_fun(as_dict=True)
    def helper(in_path):
        with open(f'{in_path}/log.time') as f:
            lines = [line for line in f
                    if(info in line)]
        raw_line=lines[-1]
        match= pattern.findall(raw_line)[-1]
        return float(match)
    time_dict=helper(in_path)
    total= sum(time_dict.values())
    print(total)
    for name_i,time_i in time_dict.items():
        print(f'{name_i}:{(time_i/total):.2f}')

if __name__ == "__main__":
   in_path='../cl/out'
   time_stats(in_path,info='HYPER')