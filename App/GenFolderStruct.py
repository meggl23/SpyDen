import os
import shutil
import tifffile as tf
import regex as re


def CreateCellDirs(SourceDir, TargetDir, Name):
    """Function that generates the folders with the correct files in the chosen target directory"""

    TargetDir = TargetDir + "/"
    SourceDir = SourceDir + "/"
    Name = Name + "/"
    os.mkdir(TargetDir + Name)

    i = 1
    regex = re.compile(".\d+")
    f = open(TargetDir + Name + "Index.txt", "a")
    for fold in sorted(os.listdir(SourceDir)):
        if os.path.isdir(SourceDir + fold):
            os.mkdir(TargetDir + Name + "cell_" + str(i))
            f.write("cell_" + str(i) + ":" + fold + "\n")
            for x in os.listdir(SourceDir + fold):
                if ("lsm" in x or "tif" in x) and re.findall(regex, x):
                    path1 = SourceDir + fold + "/" + x
                    path2 = TargetDir + Name + "cell_" + str(i) + "/" + x
                    shutil.copyfile(path1, path2)
            i += 1

    f.close()