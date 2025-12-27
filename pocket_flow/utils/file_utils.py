import os


def verify_dir_exists(dirname):
    if not os.path.isdir(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname))
