import os
import shutil
import tifffile as tf
import regex as re
import numpy as np


def CreateCellDirs(SourceDir, TargetDir, Name):
    """Function that generates the folders with the correct files in the chosen target directory"""

    AtLeastOne = 1
    TargetDir = TargetDir + "/"
    SourceDir = SourceDir + "/"
    if(not Name==""):
        Name = Name + "/"
        os.mkdir(TargetDir + Name)

    regex = re.compile(".\d+")
    f = open(TargetDir + Name + "Index.txt", "a")
    i = 1
    for fold in sorted(os.listdir(SourceDir)):
        if (os.path.isdir(SourceDir + fold) and len(os.listdir(SourceDir + fold))>0):
            tDir = os.listdir(SourceDir + fold)
            if ((np.char.find(tDir,'lsm')>-1).any() or (np.char.find(tDir,'tif')>-1).any()):
                AtLeastOne = 0
                os.mkdir(TargetDir + Name + "cell_" + str(i))
                f.write("cell_" + str(i) + ":" + fold + "\n")
                for x in os.listdir(SourceDir + fold):
                    if ("lsm" in x or "tif" in x) and re.findall(regex, x):
                        path1 = SourceDir + fold + "/" + x
                        path2 = TargetDir + Name + "cell_" + str(i) + "/" + x
                        shutil.copyfile(path1, path2)
                i+=1

    f.close()
    if(AtLeastOne==1):
        os.remove(TargetDir + Name + "Index.txt")
        if(not Name==""):
            shutil.rmtree(TargetDir + Name)
    return AtLeastOne