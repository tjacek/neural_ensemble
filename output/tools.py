import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))
import data,learn,utils

def read_results(output):
    @utils.dir_fun(as_dict=True)
    @utils.dir_fun(as_dict=True)
    def helper(path_i):
        return [ learn.read_result(path_j)
            for path_j in data.top_files(path_i)]
    return helper(output)