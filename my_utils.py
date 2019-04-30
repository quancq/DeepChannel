import shutil
import os
from datetime import datetime
import pandas as pd
import json
import numpy as np
import cv2
cv2.setNumThreads(0)
import ast

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


def load_img(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        return img
    except:
        print("Error when load image from ", img_path)
        return None


def save_img(img, img_path):
    make_parent_dirs(img_path)
    try:
        cv2.imwrite(img_path, img)
    except:
        print("Error when save img to ", img_path)


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
        if (i+1) % 10 == 0:
            print("Copying {}/{} ...".format(i+1, total_paths))
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


# def save_log(log_save_folder, log):
#     # Save log
#     # best_valid_epochs.append(dict(Epoch=best_valid_epoch, Total_Loss=min_valid_loss))
#     # log["Best_Valid_Epochs"] = best_valid_epochs
#     save_json_path = os.path.join(log_save_folder, "log.txt")
#     save_json(log, save_json_path)
#
#     save_csv_path = os.path.join(log_save_folder, "log.csv")
#     cols = ["Epoch",
#             "Train_Localization_Loss", "Train_Classification_Loss", "Train_Total_Loss", "Train_Time",
#             "Valid_Localization_Loss", "Valid_Classification_Loss", "Valid_Total_Loss", "Valid_Time"]
#     df = []
#     epoch_list = list(log["Train"]["time"].keys())
#     # epoch_ints = [int(e) for e in epoch_ints]
#     for epoch_id in sorted(epoch_list):
#         # epoch_id = str(epoch_id)
#         train_localization_loss = np.mean(log["Train"]["localization_loss"][epoch_id])
#         train_classification_loss = np.mean(log["Train"]["classification_loss"][epoch_id])
#         train_total_loss = np.mean(log["Train"]["total_loss"][epoch_id])
#         train_time = log["Train"]["time"][epoch_id]
#
#         try:
#             valid_localization_loss = np.mean(log["Valid"]["localization_loss"].get(epoch_id))
#             valid_classification_loss = np.mean(log["Valid"]["classification_loss"].get(epoch_id))
#             valid_total_loss = np.mean(log["Valid"]["total_loss"].get(epoch_id))
#             valid_time = log["Valid"]["time"].get(epoch_id)
#         except:
#             valid_localization_loss = 0
#             valid_classification_loss = 0
#             valid_total_loss = 0
#             valid_time = 0
#
#         df.append((int(epoch_id),
#                    "{:.4f}".format(train_localization_loss),
#                    "{:.4f}".format(train_classification_loss),
#                    "{:.4f}".format(train_total_loss),
#                    "{:.2f}".format(train_time),
#                    "{:.4f}".format(valid_localization_loss),
#                    "{:.4f}".format(valid_classification_loss),
#                    "{:.4f}".format(valid_total_loss),
#                    "{:.2f}".format(valid_time)
#                    ))
#
#     df.sort(key=lambda x: x[0])
#     df = pd.DataFrame(df, columns=cols)
#     save_csv(df, save_csv_path)
#
#     plot_loss(df, log_save_folder)


if __name__ == "__main__":
    pass
