import datetime
import glob
import shutil
import numpy as np


def filename_to_timestamp(filename):
    return (datetime.datetime(*map(int, filename[:-4].split("_")[1:])) -
            datetime.datetime(1970, 1, 1)).seconds


if __name__ == "__main__":
    files = glob.glob("image_*.jpg")
    timestamps = map(filename_to_timestamp, files)
    order = np.argsort(timestamps)
    for i, j in enumerate(order):
        newfile = "image%03d.jpg" % i
        print("%s -> %s" % (files[j], newfile))
        shutil.copyfile(files[j], newfile)