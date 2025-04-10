import os
import shutil
import tifffile as tf
import regex as re
import numpy as np

from .DataRead import getMetadata,CheckFiles

def CreateCellDirs(SourceDir, TargetDir, Name):
    try:
        TargetDir = TargetDir + "/"
        SourceDir = SourceDir + "/"

        if(not Name==""):
            Name = Name + "/"
            os.mkdir(TargetDir + Name)
            TargetDir = TargetDir + Name

        HasSubfolders = np.array([os.path.isdir(os.path.join(SourceDir,fold)) for fold in os.listdir(SourceDir)]).any()
        if not HasSubfolders:
            tDir = os.listdir(SourceDir)
            if ((np.char.find(tDir,'lsm')>-1).any() or (np.char.find(tDir,'tif')>-1).any() or (np.char.find(tDir,'jpg')>-1).any() or (np.char.find(tDir,'png')>-1).any()):
                FN = CheckCopyFiles(SourceDir,np.sum)
                for x in FN:
                    path1 = SourceDir + "/" + x
                    path2 = TargetDir + "/" + x
                    shutil.copyfile(path1, path2)  
        else:
            for fold in os.listdir(SourceDir):
                if (os.path.isdir(SourceDir + fold) and len(os.listdir(SourceDir + fold))>0):
                    tDir = os.listdir(SourceDir + fold)
                    if ((np.char.find(tDir,'lsm')>-1).any() or (np.char.find(tDir,'tif')>-1).any() or (np.char.find(tDir,'jpg')>-1).any() or (np.char.find(tDir,'png')>-1).any()):
                        os.mkdir(TargetDir + fold)
                        FN = CheckCopyFiles(SourceDir + fold,np.sum)
                        for x in FN:
                            path1 = SourceDir + fold + "/" + x
                            path2 = TargetDir + fold + "/" + x
                            shutil.copyfile(path1, path2)

        return 0
    except Exception as err:
        return err

def CheckCopyFiles(Dir,z_type):
    File_Names, _ = CheckFiles(Dir)
    FN = File_Names[:]
    tiff_Arr = []
    for i, x in enumerate(File_Names):
        try:
            md,temp = getMetadata(Dir + "/" + x)
            temp_mod = temp.reshape(md[1:])
            tiff_Arr.append(z_type(temp_mod, axis=0))  
        except:
            FN.remove(x)                      
    return FN