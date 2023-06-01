import sys, getopt
import argparse
import os

from tabulate import tabulate
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool
from colorama import init
from colorama import Fore, Back, Style
import h5py
import glob
import pprint
import shutil

def generate_parser():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Scan directory for SPARCSpy projects.')
    
    # Required positional argument
    parser.add_argument('search_directory', type=str,nargs='?',
                        help='directory containing SPARCSpy projects')
    
    parser.add_argument("-t","--threads", type=int, default=8, help="number of threads")
    
    parser.add_argument("-r","--recursion", type=int, default=10, help="levels of recursion")

    parser.add_argument("-n", "--dryrun", type=str, default = "True", help="Indicate if a dry run should be performed")
    
    return parser

def main():
    print("Searching for intermediate files that can be deleted, this may take a moment...")
    parser = generate_parser()
    args = parser.parse_args()
    
    global num_threads 
    num_threads= args.threads
    dry_run = args.dryrun
    
    if args.search_directory is None:
        search_directory = os.getcwd()
        
    else:
        try:
            search_directory = os.path.abspath(args.search_directory)
        except:
            print("search directory not a valid path")
        
    tabel = scan_directory_clean(args.recursion, search_directory)
    
    for line in tabel:
        dir_name, files, dirs, file_sizes, dir_sizes = line
        print("\n", dir_name, sep = "")

        if dry_run == "True":
            file_sizes = [sizeof_fmt(file) for file in file_sizes]
            dir_sizes = [sizeof_fmt(dir) for dir in dir_sizes]

            print("Found the following files to delete:")
            print(*zip(files, file_sizes), sep = "\n")
            print("Found the following directories to delete:")
            print(*zip(dirs, dir_sizes), sep = "\n")
            print('Rerun with -n False to remove these files')

        if dry_run == "False":
            print("Deleting files...")

            file_sizes = sum(file_sizes)
            dir_sizes = sum(dir_sizes)

            with Pool(max_workers=min(5, num_threads)) as pool:
                pool.map(shutil.rmtree, dirs)
                pool.map(os.remove, files)

            print("Deleted files with a total storage size of", sizeof_fmt(file_sizes + dir_sizes))

def scan_directory_clean(levels_left, path, num_threads = 10):

    if levels_left > 0:

        config_name = "config.yml"
        is_project_dir = os.path.isfile(os.path.join(path,config_name))

        if is_project_dir: 

            dir_name = os.path.basename(path)

            _to_delete = []
            _to_delete = _to_delete + glob.glob(os.path.join(path, "segmentation", "tiles", "*"))
            _to_delete = _to_delete + glob.glob(os.path.join(path, "segmentation", "shards", "*")) #for backward compatability with previous runs where it was still called shards
            #_to_delete.append(os.path.join(path, 'segmentation', 'input_image.h5'))

            files = []
            dirs = []

            for _path in _to_delete:
                
                if os.path.isdir(_path):
                    dirs.append(_path)
                
                elif os.path.exists(_path):
                    files.append(_path)
                else:
                    #print(_path)
                    continue

            file_sizes = [] 
            dir_sizes = []

            for file in files:
                file_sizes.append(get_file_size(file))
            for dir in dirs:
                dir_sizes.append(get_dir_size(dir))

            return([[dir_name, files, dirs, file_sizes, dir_sizes]])

        else:
            #iterate all subfolders
            current_level_directories = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

            with Pool(max_workers=num_threads) as pool:
                projects = pool.map(partial(scan_directory_clean, levels_left-1), current_level_directories)
            
            return list(flatten(projects))

def check_dir(path):
    _files = []
    _dirs = []

    if os.path.isdir(path):
        _list = glob.glob(os.path.join(path,'*'))
        #print(_list)
        
        for _file in _list:
            #_file = os.path.join(path, _file)
            #print(_file)
            if os.path.isdir(_file):
                _dirs.append(_file)
            elif os.path.isfile(_file):
                _files.append(_file)
            
    if os.path.isfile(path):
        _files.append(file)
    
    return(_files, _dirs)


def get_dir_size(path):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total

def get_file_size(path):
    if os.path.isfile(path):
        return os.stat(path).st_size
    else:
        return None

# https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

from collections.abc import Iterable
def flatten(l):
    for el in l:
        if isinstance(el, Iterable):
            if len(el)>0:
                yield el[0]
        else:
            # pool.map might return None on subfolders
            if el is not None:
                yield el

if __name__ == "__main__":
    main()