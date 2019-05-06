import shutil
import os
from datetime import datetime
import pandas as pd
import json
import numpy as np
import ast
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def get_dir_paths(parent_dir):
    dir_paths = [os.path.join(parent_dir, dir) for dir in os.listdir(parent_dir)]
    dir_paths = [dir for dir in dir_paths if os.path.isdir(dir)]
    return dir_paths


def get_dir_names(parent_dir):
    dir_names = [dir_name for dir_name in os.listdir(parent_dir)
                 if os.path.isdir(os.path.join(parent_dir, dir_name))]
    return dir_names


def get_dir_name_of_path(path):
    return os.path.basename(os.path.dirname(path))


def get_file_names(parent_dir):
    file_names = [file_name for file_name in os.listdir(parent_dir)
                  if os.path.isfile(os.path.join(parent_dir, file_name))]
    return file_names


def get_file_paths(parent_dir):
    file_paths = [os.path.join(parent_dir, file_name) for file_name in os.listdir(parent_dir)
                  if os.path.isfile(os.path.join(parent_dir, file_name))]
    return file_paths


def get_parent_path(path):
    return path[:path.rfind("/")]


def get_all_file_paths(dir, abs_path=False):
    file_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            if abs_path:
                path = os.path.abspath(path)
            file_paths.append(path)

    return file_paths


def get_files_with_extension(paths, extensions):
    result = []
    for path in paths:
        for ext in extensions:
            if path.endswith(ext):
                result.append(path)
                break
    return result


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def make_parent_dirs(path):
    dir = get_parent_path(path)
    make_dirs(dir)


def load_str(path):
    data = ""
    try:
        with open(path, 'r') as f:
            data = f.read().strip()
    except:
        print("Error when load str from ", os.path.abspath(path))

    return data


def save_str(data, save_path):
    make_parent_dirs(save_path)
    try:
        with open(save_path, 'w') as f:
            f.write(data)
        print("Save str data to {} done".format(save_path))
    except:
        print("Error when save str to ", os.path.abspath(save_path))


def get_time_str(time=datetime.now(), fmt=DEFAULT_TIME_FORMAT):
    try:
        return time.strftime(fmt)
    except:
        return ""


def save_list(lst, save_path):
    make_parent_dirs(save_path)

    with open(save_path, "w") as f:
        f.write("\n".join(lst))

    print("Save data (size = {}) to {} done".format(len(lst), save_path))


def load_list(path):
    data = []
    with open(path, 'r') as f:
        data = f.read().strip().split("\n")

    print("Load list data (size = {}) from {} done".format(len(data), path))
    return data


def load_csv(path, **kwargs):
    data = None
    try:
        data = pd.read_csv(path, **kwargs)
        print("Read csv data (size = {}) from {} done".format(data.shape[0], path))
    except Exception as e:
        print("Error {} when load csv data from {}".format(e, path))
    return data


def save_csv(df, path, fields=None, **kwargs):
    make_parent_dirs(path)
    if fields is not None:
        df = df[fields]
    df.to_csv(path, index=False, **kwargs)
    print("Save csv data (size = {}) to {} done".format(df.shape[0], path))


def save_json(data, path):
    make_parent_dirs(path)
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, default=MyEncoder)
    print("Save json data (size = {}) to {} done".format(len(data), path))


def load_json(path):
    data = {}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception:
        print("Error when load json from ", path)

    return data


def load_json_lines(path):
    data = load_list(path)
    data = [ast.literal_eval(elm) for elm in data]

    return data


def copy_file(src_path, dst_path):
    try:
        make_parent_dirs(dst_path)
        shutil.copyfile(src_path, dst_path)
        # print("Copy file from {} to {} done".format(src_path, dst_path))
        return True
    except Exception:
        print("Error: when copy file from {} to {}".format(src_path, dst_path))
        return False


def copy_files(src_dst_paths):
    total_paths = len(src_dst_paths)
    num_success = 0
    for i, (src_path, dst_path) in enumerate(src_dst_paths):
        if (i + 1) % 10 == 0:
            print("Copying {}/{} ...".format(i + 1, total_paths))
        is_success = copy_file(src_path, dst_path)
        if is_success:
            num_success += 1

    print("Copy {}/{} files done".format(num_success, total_paths))
    return num_success


def move_file(src_path, dst_path):
    try:
        make_parent_dirs(dst_path)
        shutil.move(src_path, dst_path)
        # print("Move file from {} to {} done".format(src_path, dst_path))
        return True
    except Exception:
        print("Error: when move file from {} to {}".format(src_path, dst_path))
        return False


def convert_glove_to_word2vec(glove_path, save_path):
    glove2word2vec(glove_input_file=glove_path, word2vec_output_file=save_path)
    print("Convert glove -> word2vec done. Save word2vec to ", save_path)


def load_we_gensim(path):
    glove = KeyedVectors.load_word2vec_format(path, binary=False)
    return glove


def convert_word2vec_bin_to_text(bin_path, txt_path):
    model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
    model.save_word2vec_format(txt_path, binary=False)
    print("Convert word2vec bin -> txt done. Save word2vec to ", txt_path)


if __name__ == "__main__":
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_path', required=True, type=str, help='word2vec bin path')
    parser.add_argument('--txt_path', required=True, type=str, help='word2vec txt path')
    args = parser.parse_args()

    convert_word2vec_bin_to_text(args.bin_path, args.txt_path)
