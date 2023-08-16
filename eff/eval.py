import tools
tools.silence_warnings()
import os,argparse

def all_files(in_path):
    for folder, subfolders, files in os.walk(in_path):
        for file_i in files:
            path_i = os.path.abspath(os.path.join(folder, file_i))
            yield path_i

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default=f'pred')
    args = parser.parse_args()
    paths=all_files(args.pred)
    print(list(paths))