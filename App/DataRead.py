from skimage.registration import phase_cross_correlation
from skimage.draw import polygon
from skimage.measure import regionprops
import imageio.v2 as imageio
import tifffile as tf
import math
import re

from .Utility import *
from .Spine    import *
from .Dendrite import *
from .PathFinding import GetLength
from .SynapseFuncs import *
from .MPL_Widget import *

import json

from PyQt5.QtCore import QCoreApplication
from matplotlib.path import Path


def Measure_BG(tiff_Arr_m, FileLen, z_type):

    """
    Input:
            tiff_Arr_m (np.array of doubles): Pixel values of all the tiff files
            FileLen                         : Number of files
            NaNlist                         : Entries where the correct file is not available
    Output:
            bg_list (np.array of doubles): values of background

    Function:
            Finds 4 corners of image and works out average, using this as background
            and kicks out any values which are 2 x the others
    """

    width = 20
    pt1 = [20, 20]
    if FileLen > 1:
        bg_list = []
        for i in range(FileLen):
            bgMeasurement1 = []
            bgMeasurement2 = []
            bgMeasurement3 = []
            bgMeasurement4 = []

            for ii in range(20 + width):
                for jj in range(20 + width):
                    if ((ii - pt1[0]) ** 2 + (jj - pt1[1]) ** 2) < width**2:
                        bgMeasurement1.append(tiff_Arr_m[i, ii, jj])
                        bgMeasurement2.append(
                            tiff_Arr_m[i, ii, tiff_Arr_m.shape[-1] - jj]
                        )
                        bgMeasurement3.append(
                            tiff_Arr_m[i, tiff_Arr_m.shape[-2] - ii, jj]
                        )
                        bgMeasurement4.append(
                            tiff_Arr_m[
                                i, tiff_Arr_m.shape[-2] - ii, tiff_Arr_m.shape[-1] - jj
                            ]
                        )

            bg = np.array(
                [
                    np.mean(bgMeasurement1),
                    np.mean(bgMeasurement2),
                    np.mean(bgMeasurement3),
                    np.mean(bgMeasurement4),
                ]
            )
            bg = np.array(bg.min())

            bg_list.append(bg.min())

        return bg_list
    else:
        bgMeasurement1 = []
        bgMeasurement2 = []
        bgMeasurement3 = []
        bgMeasurement4 = []

        for ii in range(20 + width):
            for jj in range(20 + width):
                if ((ii - pt1[0]) ** 2 + (jj - pt1[1]) ** 2) < width**2:
                    bgMeasurement1.append(tiff_Arr_m[0, ii, jj])
                    bgMeasurement2.append(tiff_Arr_m[0, ii, tiff_Arr_m.shape[-1] - jj])
                    bgMeasurement3.append(tiff_Arr_m[0, tiff_Arr_m.shape[-2] - ii, jj])
                    bgMeasurement4.append(
                        tiff_Arr_m[
                            0, tiff_Arr_m.shape[-2] - ii, tiff_Arr_m.shape[-1] - jj
                        ]
                    )

        bg = np.array(
            [
                np.mean(bgMeasurement1),
                np.mean(bgMeasurement2),
                np.mean(bgMeasurement3),
                np.mean(bgMeasurement4),
            ]
        )
        bg = np.array(bg.min())

        return bg


def GetTiffData(File_Names, scale, z_type=np.sum, Dir=None, Channels=False):

    """
    Input:
            File_Names (array of Strings): Holding name of timesteps
            scale (double)               : Pixel to μm?
            Dir (String)                 : Super directory we are looking at
            zStack (Bool)                : Flag wether we are looking at zstacks
            as_gray (Bool)               : Flag wether we want grayscale or not

    Output:
            tiff_Arr (np.array of doubles): Pixel values of all the tiff files

    Function:
            Uses tiff library to get values
    """

    Times = []

    if File_Names == None:
        File_Names, Times = CheckFiles(Dir)

    if File_Names[0].endswith(".lsm"):
        scale = getScale(Dir + "/" + File_Names[0])
    else:
        scale = scale

    tiff_Arr = []
    for i, x in enumerate(File_Names):
        md,temp = getMetadata(Dir + "/" + x)
        temp_mod = temp.reshape(md[1:])
        if not Channels:
            temp_mod = z_type(temp_mod, axis=1, keepdims=True)
        tiff_Arr.append(z_type(temp_mod, axis=0))

    md[0] = len(tiff_Arr)
    if not z_type == None:
        md[1] = 1
    md[2:] = tiff_Arr[0].shape
    tiff_Arr = np.array(tiff_Arr)

    if z_type == np.sum:

        tiff_Arr = 256*tiff_Arr/tiff_Arr.max() # dividing by the max to get it back within RGB range. Dividing by the max of everything because if image 1 has a lower sum then the normalised version should too.

    return tiff_Arr, Times, md, scale


def getMetadata(filename, frame=None):

    """
    Input:
            filename (string) : Name of file to be read

    Output:
            meta_data (int)   : File MetaData

    Function:
            Get meta_data such as dims from file
    """

    if filename.endswith(".tif"):
        return getTifDimensions(filename),tf.imread(filename)
    elif filename.endswith(".lsm"):
        return getLSMDimensions(filename),tf.imread(filename)
    elif(filename.endswith('.png') or filename.endswith('.jpg')):
        temp = imageio.imread(filename)
        if(len(temp.shape)>2):
            temp = np.dot(temp[..., :3], [0.2989, 0.5870, 0.1140])
        meta_data = np.ones((5))
        meta_data[1] = 1
        meta_data[2] = 1
        meta_data[0] = 1
        meta_data[3] = temp.shape[-2]
        meta_data[4] = temp.shape[-1]
        return meta_data.astype(int),temp
    else:
        if frame is None:
            print("Unsupported file format found. contact admin")
        # TODO: Format print as pop-up/In the main window

def getScale(filename):
    tf_file = tf.TiffFile(filename)
    if filename.endswith(".tif"):
        return 0.114
    elif filename.endswith(".lsm"):
        return tf_file.lsm_metadata["ScanInformation"]["SampleSpacing"]
    else:
        print("Unsupported file format found. contact admin")

def getTifDimensions(filename):

    """
    Input:
            filename (string) : Name of file to be read

    Output:
            meta_data (int)   : File MetaData

    Function:
            Get meta_data such as dims from tif file
    """
    try:
        meta_data = np.ones((5))  # to hold # of (t,z,c,y,x)
        tf_file = tf.TiffFile(filename)

        if "slices" in tf_file.imagej_metadata.keys():
            meta_data[1] = tf_file.imagej_metadata["slices"]
        if "channels" in tf_file.imagej_metadata.keys():
            meta_data[2] = tf_file.imagej_metadata["channels"]
        if "time" in tf_file.imagej_metadata.keys():
            meta_data[0] = tf_file.imagej_metadata["time"]

        d = tf_file.asarray()
        meta_data[3] = d.shape[-2]
        meta_data[4] = d.shape[-1]
    except:
        temp = tf.imread(filename)
        meta_data[1] = temp.shape[0]
        meta_data[2] = 1
        meta_data[0] = 1
        meta_data[3] = temp.shape[-2]
        meta_data[4] = temp.shape[-1]

    return meta_data.astype(int)


def getLSMDimensions(filename):

    """
    Input:
            filename (string) : Name of file to be read

    Output:
            meta_data (int)   : File MetaData

    Function:
            Get meta_data such as dims from lsm file
    """

    meta_data = np.ones((5))
    lsm_file = tf.TiffFile(filename)

    meta_data[0] = lsm_file.lsm_metadata["DimensionTime"]
    meta_data[1] = lsm_file.lsm_metadata["DimensionZ"]
    meta_data[2] = lsm_file.lsm_metadata["DimensionChannels"]
    meta_data[3] = lsm_file.lsm_metadata["DimensionY"]
    meta_data[4] = lsm_file.lsm_metadata["DimensionX"]

    return meta_data.astype(int)


def CheckFiles(Dir):

    """
    Input:
            Dir (String)                 : Super directory we are looking at

    Output:
            Time (list of strings)  : Available files in directory

    Function:
            Checks if files ending with tif or lsm are in the folder and then augments
            the list of files with necessary ones
    """


    File_Names = []
    for x in os.listdir(Dir):
        if ".lsm" in x or ".tif" in x or ".png" in x or ".jpg" in x:
            File_Names.append(x)
    if any(entry.endswith(('lsm', 'tif')) for entry in File_Names):
        File_Names = [entry for entry in File_Names if not entry.endswith(('png', 'jpg'))]
    
    try:
        try:
            regex = re.compile(".\d+")
            File_Names_int = [re.findall(regex, f)[0] for f in File_Names]
        except:
            regex = re.compile("\d+")
            File_Names_int = [re.findall(regex, f)[0] for f in File_Names]
    except:
        File_Names_int = np.arange(0,len(File_Names))

    try:
        try:
            File_Names_int = [int(f) for f in File_Names_int]
        except:
            File_Names_int = [int(f[1:]) for f in File_Names_int]
        File_Names = [x for _, x in sorted(zip(File_Names_int, File_Names))]

    except:
        pass

    File_Names_int.sort()

    return File_Names, File_Names_int


def GetTiffShift(tiff_Arr, SimVars):

    """
    Input:
            tiff_Arr (np.array) : The pixel values of the of tiff files
            SimVars  (class)    : The class holding all simulation parameters

    Output:
            tiff_arr (np.array) : The shift tiff_arr so that all snapshots overlap

    Function:
            Does an exhaustive search to find the best fitting shift and then applies
            the shift to the tiff_arr
    """

    Dir = SimVars.Dir

    nSnaps = SimVars.Snapshots
    if os.path.isfile(Dir + "MinDir.npy") == True:
        MinDirCum = np.load(Dir + "MinDir.npy")
    else:
        MinDir = np.zeros([2, nSnaps - 1])
        if not (SimVars.frame == None):
            SimVars.frame.set_status_message.setText('Computing overlap vector')
        for t in range(nSnaps - 1):
            shift, _, _ = phase_cross_correlation(
                tiff_Arr[t, 0, :, :], tiff_Arr[t + 1, 0, :, :]
            )
            MinDir[:, t] = -shift
            SimVars.frame.set_status_message.setText(SimVars.frame.set_status_message.text()+'.')
            QCoreApplication.processEvents()
            SimVars.frame.set_status_message.repaint()

        MinDirCum = np.cumsum(MinDir, 1)
        MinDirCum = np.insert(MinDirCum, 0, 0, 1)
        np.save(Dir + "MinDir.npy", MinDirCum)

    MinDirCum = MinDirCum.astype(int)
    tf,SimVars.xLimsG,SimVars.yLimsG = ShiftArr(tiff_Arr, MinDirCum)
    return tf

def GetTiffShiftDend(tiff_Arr, SimVars,dX,dY):

    """
    Input:
            tiff_Arr (np.array) : The pixel values of the of tiff files
            SimVars  (class)    : The class holding all simulation parameters

    Output:
            tiff_arr (np.array) : The shift tiff_arr so that all snapshots overlap

    Function:
            Does an exhaustive search to find the best fitting shift and then applies
            the shift to the tiff_arr
    """

    Dir = SimVars.Dir

    nSnaps = SimVars.Snapshots
    if os.path.isfile(Dir + "MinDirD.npy") == True:
        MinDirCum = np.load(Dir + "MinDirD.npy")
    else:
        MinDir = np.zeros([2, nSnaps - 1])
        if not (SimVars.frame == None):
            SimVars.frame.set_status_message.setText('Computing overlap vector')
        for t in range(nSnaps - 1):
            shift, _, _ = phase_cross_correlation(
                tiff_Arr[t, 0, dY[0]:dY[1], dX[0]:dX[1]], tiff_Arr[t + 1, 0, dY[0]:dY[1], dX[0]:dX[1]]
            )
            MinDir[:, t] = -shift
            SimVars.frame.set_status_message.setText(SimVars.frame.set_status_message.text()+'.')
            QCoreApplication.processEvents()
            SimVars.frame.set_status_message.repaint()

        MinDirCum = np.cumsum(MinDir, 1)
        MinDirCum = np.insert(MinDirCum, 0, 0, 1)
        np.save(Dir + "MinDirD.npy", MinDirCum)

    MinDirCum = MinDirCum.astype(int)
    tf,SimVars.xLimsD,SimVars.yLimsD = ShiftArr(tiff_Arr, MinDirCum)

    return tf


def ShiftArr(tiff_Arr, MinDirCum):

    """
    Input:
            tiff_Arr  (np.array)    : The pixel values of the of tiff files
            MinDirCum (np.array)    : The shifting directions

    Output:
            tiff_arr (np.array) : The shift tiff_arr so that all snapshots overlap

    Function:
            Application of MinDirCum to tiff_Arr
    """

    xLim = [(np.min(MinDirCum, 1)[0] - 1), (np.max(MinDirCum, 1)[0] + 1)]
    yLim = [(np.min(MinDirCum, 1)[1] - 1), (np.max(MinDirCum, 1)[1] + 1)]

    tiff_Arr_m = np.array(
        [
            tiff_Arr[
                i,
                :,
                -xLim[0] + MinDirCum[0, i] : -xLim[1] + MinDirCum[0, i],
                -yLim[0] + MinDirCum[1, i] : -yLim[1] + MinDirCum[1, i],
            ]
            for i in range(tiff_Arr.shape[0])
        ]
    )

    return tiff_Arr_m,xLim,yLim


def Measure(SynArr, tiff_Arr, SimVars,frame=None):

    """
    Input:
            SynArr  (list of synapses)
            tiff_Arr  (np.array)    : The pixel values of the of tiff files
            MinDirCum (np.array)    : The shifting directions

    Output:
            None

    Function:
            Function to decide if we should apply the circular measure or the
            shape measure
    """
    if(SimVars.multitime_flag):
        Snaps = SimVars.Snapshots
    else:
        Snaps = 1
    if(SimVars.multiwindow_flag):
        Chans = SimVars.Channels
    else:
        Chans = 1
    if(Chans>1):
        if(SimVars.Mode=="Luminosity"):
            for S in SynArr:
                frame.set_status_message.setText(frame.set_status_message.text()+'.')
                for i in range(SimVars.Channels):
                    Mean,Area,Max,Min,RawIntDen,IntDen,local_bg = MeasureShape_and_BG(S, tiff_Arr[:,i,:,:], SimVars,Snaps)
                    S.max.append(Max)
                    S.min.append(Min)
                    S.RawIntDen.append(RawIntDen)
                    S.IntDen.append(IntDen)
                    S.mean.append(Mean)
                    S.local_bg.append(local_bg)
                S.area.append(Area[0])
                SpineBoundingBox(S,SimVars.Unit,SimVars.Mode,Snaps = 1)

        else:
            for S in SynArr:
                frame.set_status_message.setText(frame.set_status_message.text()+'.')
                for i in range(SimVars.Channels):
                    Mean,Area,Max,Min,RawIntDen,IntDen = MeasureShape(S, tiff_Arr[:,i,:,:], SimVars,Snaps)
                    S.max.append(Max)
                    S.min.append(Min)
                    S.RawIntDen.append(RawIntDen)
                    S.IntDen.append(IntDen)
                    S.mean.append(Mean)
                    S.area.append(Area)
                SpineBoundingBox(S,SimVars.Unit,SimVars.Mode,Snaps = Snaps)
    else:
        if(SimVars.Mode=="Luminosity"):
            for S in SynArr:
                frame.set_status_message.setText(frame.set_status_message.text()+'.')
                Mean,Area,Max,Min,RawIntDen,IntDen,local_bg = MeasureShape_and_BG(S, tiff_Arr[:,SimVars.frame.actual_channel,:,:], SimVars,Snaps)
                S.max.append(Max)
                S.min.append(Min)
                S.RawIntDen.append(RawIntDen)
                S.IntDen.append(IntDen)
                S.mean.append(Mean)
                S.local_bg.append(local_bg)
                S.area.append(Area[0])
                SpineBoundingBox(S,SimVars.Unit,SimVars.Mode,Snaps = 1)
        else:
            for S in SynArr:
                frame.set_status_message.setText(frame.set_status_message.text()+'.')
                Mean,Area,Max,Min,RawIntDen,IntDen = MeasureShape(S, tiff_Arr[:,SimVars.frame.actual_channel,:,:], SimVars,Snaps)
                S.max.append(Max)
                S.min.append(Min)
                S.RawIntDen.append(RawIntDen)
                S.IntDen.append(IntDen)
                S.mean.append(Mean)
                S.area.append(Area)
                SpineBoundingBox(S,SimVars.Unit,SimVars.Mode,Snaps = Snaps)



    # We are going to exlude the part of the spine neck thats inside the ROI first before doing any further measurements:
    # - if there is only one snap we can just directly compare the ROI with the spine neck
    # - if we have local-shifts then each timestep is a different spine neck that needs to computer
    # - if we have areas then we have additionally a separate area
    NewNecks = []
    if(Snaps==1):
        for S in SynArr:
            if(S.type < 2):
                try:
                    if(np.array(S.neck).ndim>2):
                        Intersection_pt, seg_idx = find_intersection(S.neck[0], S.points[0])
                        new_neck =np.vstack([Intersection_pt, S.neck[0][seg_idx+1:]])[np.newaxis,:]
                    else:
                        Intersection_pt, seg_idx = find_intersection(S.neck, S.points)
                        new_neck =np.vstack([Intersection_pt, S.neck[seg_idx+1:]])[np.newaxis,:]
                except:
                    new_neck = np.array(S.neck)


                S.neck_length = [GetLength(new_neck)*SimVars.Unit]
                NewNecks.append(new_neck)
            else:
                S.neck_length = [0]
                NewNecks.append([])
    elif(SimVars.Mode=="Luminosity"):
        if(SimVars.frame.local_shift):
            
            for S in SynArr:
                if(S.type < 2):
                    nn = []
                    for n,s in zip(S.neck,S.shift):
                        try:
                            Intersection_pt, seg_idx = find_intersection(n, np.array(S.points)+ [s[0],s[1]])
                            new_neck =np.vstack([Intersection_pt, n[seg_idx+1:]])
                        except:
                            new_neck = np.array(n) 
                        S.neck_length.append(GetLength(new_neck)*SimVars.Unit)
                        nn.append(new_neck)

                    NewNecks.append(nn)
                else:
                    S.neck_length = [0]
                    NewNecks.append([])
        else:   
            for S in SynArr:
                if(S.type < 2):
                    try:
                        Intersection_pt, seg_idx = find_intersection(S.neck, S.points)
                        new_neck =np.vstack([Intersection_pt, S.neck[seg_idx+1:]])
                    except:
                        new_neck = np.array(S.neck)
                    S.neck_length = [GetLength(new_neck)*SimVars.Unit]
                    NewNecks.append(new_neck)
                else:
                    S.neck_length = [0]
                    NewNecks.append([])
    elif(SimVars.Mode=="Area"):
        if(SimVars.frame.local_shift):
            for S in SynArr:
                if(S.type < 2):
                    nn = []
                    for n,s in zip(S.neck,S.shift):
                        try:
                            Intersection_pt, seg_idx = find_intersection(n, np.array(S.points)+ [s[0],s[1]])
                            new_neck =np.vstack([Intersection_pt, n[seg_idx+1:]])
                        except:
                            new_neck = np.array(n) 
                        S.neck_length.append(GetLength(new_neck)*SimVars.Unit)
                        nn.append(new_neck)

                    NewNecks.append(nn)
                else:
                    S.neck_length = [0]
                    NewNecks.append([])
        else:
            for S in SynArr:
                nn = []
                if(S.type < 2):
                    for n,p in zip(S.neck,S.points):
                        try:
                            Intersection_pt, seg_idx = find_intersection(n, p)
                            new_neck =np.vstack([Intersection_pt, n[seg_idx+1:]])
                        except:
                            new_neck = np.array(n)
                        S.neck_length.append(GetLength(new_neck)*SimVars.Unit)
                        nn.append(new_neck)
                    NewNecks.append(nn)
                else:
                    S.neck_length = [0]
                    NewNecks.append([])


    neck_factor ="{:.1f}".format(frame.spine_neck_width_mult_slider.value()*0.1)
    frame.spine_neck_width_mult_counter.setText(neck_factor)
    frame.spine_neck_sigma_counter.setText(str(frame.spine_neck_sigma_slider.value()))

    try:
        if hasattr(frame,"ContourLines"):
            for l in frame.ContourLines:
                try:
                    l.remove()
                except:
                    pass
            del frame.ContourLines
    except Exception as e:
        pass

    frame.ContourLines = []
    for N,S in zip(NewNecks,SynArr):
        if((SimVars.Mode == "Luminosity" and frame.local_shift) or (SimVars.Mode == "Area" and SimVars.multitime_flag)):
            if(S.type < 2):
                Contours = []
                AvgWidth = []
                AvgLum   = []
                for i,n in enumerate(N):
                    n = np.round(n).astype(int)
                    bbmin = (max(np.min(n[:,1]) - 50, 0),max(np.min(n[:,0]) - 50, 0))
                    bbmax = (min(np.max(n[:,1]) + 50, tiff_Arr.shape[-2]),min(np.max(n[:,0]) + 50, tiff_Arr.shape[-1]))
                    n_shift = n-bbmin[::-1]
                    tiff_Arr_small = tiff_Arr[i,frame.actual_channel,bbmin[0]:bbmax[0], bbmin[1]:bbmax[1]]
                    c,w = FindNeckWidth(n_shift,tiff_Arr_small,S.neck_thresh[i],sigma = frame.spine_neck_sigma_slider.value(),width_factor = frame.spine_neck_width_mult_slider.value()*0.1)
                    contour = c[0] + bbmin[::-1]
                    AvgWidth.append(w*SimVars.Unit)
                    Contours.append(contour.squeeze())
                S.neck_contours = Contours
                S.neck_mean  = np.array([Luminosity_from_contour(c,tiff_Arr[i]) for i,c in enumerate(S.neck_contours)]).T.tolist()
                S.neck_width = AvgWidth
                line, = plt.plot(S.neck_contours[frame.actual_timestep][:, 0], S.neck_contours[frame.actual_timestep][:, 1], 'y')
                frame.ContourLines.append(line)
            else:
                S.neck_contours,S.neck_mean,S.neck_width = [],[],[]
                frame.ContourLines.append([])
        else:
            if(S.type < 2):
                N = np.round(N).astype(int)
                if(N.ndim > 2):
                    N = N[0]
                bbmin = (max(np.min(N[:,1]) - 50, 0),max(np.min(N[:,0]) - 50, 0))
                bbmax = (min(np.max(N[:,1]) + 50, tiff_Arr.shape[-2]),min(np.max(N[:,0]) + 50, tiff_Arr.shape[-1]))
                N_shift = N-bbmin[::-1]
                tiff_Arr_small = tiff_Arr[frame.actual_timestep,frame.actual_channel,bbmin[0]:bbmax[0], bbmin[1]:bbmax[1]]

                c,w = FindNeckWidth(N_shift,tiff_Arr_small,S.neck_thresh,sigma = frame.spine_neck_sigma_slider.value(),width_factor = frame.spine_neck_width_mult_slider.value()*0.1)
                S.neck_contours = (c[0] + bbmin[::-1]).squeeze()
                S.neck_width    = [w*SimVars.Unit]
                if(SimVars.multitime_flag):
                    S.neck_mean  = np.array([Luminosity_from_contour(S.neck_contours,tiff_Arr[i]) for i in range(SimVars.Snapshots)]).T.tolist()
                else:
                    S.neck_mean  = np.array([Luminosity_from_contour(S.neck_contours,tiff_Arr[frame.actual_timestep])]).T.tolist()
                line, = plt.plot(S.neck_contours[:, 0], S.neck_contours[:, 1], 'y')
                frame.ContourLines.append(line)
            else:
                S.neck_contours,S.neck_mean,S.neck_width = [],[],[]
                frame.ContourLines.append([])
def Luminosity_from_contour(contour,image):

    height,width = image.shape[1:]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.vstack((x.flatten(), y.flatten())).T

    path_contour = Path(contour)

    mask_flat = path_contour.contains_points(points)
    mask = mask_flat.reshape((height, width))
    return image[:,mask].mean(axis=-1)


def MeasureShape(S, tiff_Arr, SimVars,Snapshots):

    """
    Input:
            S (Synapse)
            tiff_Arr (np.array) : The pixel values of the of tiff files
            SimVars  (class)    : The class holding all simulation parameters
    Output:
            None

    Function:
            Finds the relevant places in the tiff file and measures these for each synapse
    """
    SynA = S.points

    Mean = []
    area = []
    Max  = []
    Min  = []
    RawIntDen = []
    IntDen = []
    if(np.array(SynA).ndim == 3):
        for i in range(Snapshots):
            try:
                SynL = np.array(SynA[i]) + S.shift[i]
            except:
                SynL = np.array(SynA[i])
            SynL[:,0] = np.clip(SynL[:,0],0,tiff_Arr.shape[-1]-1)
            SynL[:,1] = np.clip(SynL[:,1],0,tiff_Arr.shape[-2]-1)
            
            if SynL.ndim == 2:
                mask = np.zeros(shape=tiff_Arr.shape[-2:], dtype=np.uint8)
                c = SynL[:, 1]
                r = SynL[:, 0]
                rr, cc = polygon(r, c)
                mask[cc, rr] = 1
                props = regionprops(mask.astype(int))
                try:
                    roi  = tiff_Arr[i].astype(np.float64)
                    roi[np.where(mask == 0)] = math.nan
                    area_pix = np.sum(mask)
                    area.append(int(area_pix) * SimVars.Unit**2)
                    Max.append(int(np.nanmax(roi)))
                    Min.append(int(np.nanmin(roi)))
                    RawIntDen.append(int(np.nansum(roi)))
                    IntDen.append(np.nansum(roi) * SimVars.Unit**2)
                    Mean.append(np.nanmean(roi))

                except Exception as ex:
                    area.append(math.nan)
                    Mean.append(math.nan)
                    Max.append(math.nan)
                    Min.append(math.nan)
                    RawIntDen.append(math.nan)
                    IntDen.append(math.nan)
    else:
        SynL = np.array(SynA)
        SynL[:,0] = np.clip(SynL[:,0],0,tiff_Arr.shape[-1]-1)
        SynL[:,1] = np.clip(SynL[:,1],0,tiff_Arr.shape[-2]-1)
        
        if SynL.ndim == 2:
            mask = np.zeros(shape=tiff_Arr.shape[-2:], dtype=np.uint8)
            c = SynL[:, 1]
            r = SynL[:, 0]
            rr, cc = polygon(r, c)
            mask[cc, rr] = 1
            props = regionprops(mask.astype(int))
            try:
                roi  = tiff_Arr[i].astype(np.float64)
                roi[np.where(mask == 0)] = math.nan
                area_pix = np.sum(mask)
                area.append(int(area_pix) * SimVars.Unit**2)
                Max.append(int(np.nanmax(roi)))
                Min.append(int(np.nanmin(roi)))
                RawIntDen.append(int(np.nansum(roi)))
                IntDen.append(np.nansum(roi) * SimVars.Unit**2)
                Mean.append(np.nanmean(roi))

            except Exception as ex:
                area.append(math.nan)
                Mean.append(math.nan)
                Max.append(math.nan)
                Min.append(math.nan)
                RawIntDen.append(math.nan)
                IntDen.append(math.nan)



    return Mean,area,Max,Min,RawIntDen,IntDen

def MeasureShape_and_BG(S, tiff_Arr, SimVars, Snapshots):

    """
    Input:
            S (Synapse)
            tiff_Arr (np.array) : The pixel values of the of tiff files
            SimVars  (class)    : The class holding all simulation parameters
    Output:
            None

    Function:
            Finds the relevant places in the tiff file and measures these for each synapse
    """
    SynA = np.array(S.points)
    Mean = []
    area = []
    Max  = []
    Min  = []
    RawIntDen = []
    IntDen = []
    local_bg = []
    for i in range(Snapshots):
        try:
            SynL = SynA + S.shift[i]
            SynBg = SynA-np.array(S.location)+np.array(S.bgloc)
        except:
            SynL = SynA
            SynBg = SynA-np.array(S.location)+np.array(S.bgloc)

        SynBg[:,0] = np.clip(SynBg[:,0],0,tiff_Arr.shape[-1]-1)
        SynBg[:,1] = np.clip(SynBg[:,1],0,tiff_Arr.shape[-2]-1)
        SynL[:,0] = np.clip(SynL[:,0],0,tiff_Arr.shape[-1]-1)
        SynL[:,1] = np.clip(SynL[:,1],0,tiff_Arr.shape[-2]-1)

        if SynL.ndim == 2:
            mask = np.zeros(shape=tiff_Arr.shape[-2:], dtype=np.uint8)
            mask2 = np.zeros(shape=tiff_Arr.shape[-2:], dtype=np.uint8)

            c = SynL[:, 1]
            r = SynL[:, 0]
            rr, cc = polygon(r, c)
            mask[cc, rr] = 1
            props = regionprops(mask.astype(int))
            c = SynBg[:,1]
            r = SynBg[:,0]
            rr, cc = polygon(r, c)
            mask2[cc, rr] = 1

            try:
                roi  = tiff_Arr[i].astype(np.float64)
                roi2 = tiff_Arr[i].astype(np.float64)
                roi[np.where(mask == 0)] = math.nan
                roi2[np.where(mask2== 0)] = math.nan
                area_pix = np.sum(mask)
                area.append(int(area_pix) * SimVars.Unit**2)
                Max.append(int(np.nanmax(roi)))
                Min.append(int(np.nanmin(roi)))
                RawIntDen.append(int(np.nansum(roi)))
                IntDen.append(np.nansum(roi) * SimVars.Unit**2)
                Mean.append(np.nanmean(roi))
                local_bg.append(np.nanmean(roi2))


            except Exception as ex:
                print(ex)
                area.append(math.nan)
                Mean.append(math.nan)
                Max.append(math.nan)
                Min.append(math.nan)
                RawIntDen.append(math.nan)
                IntDen.append(math.nan)
                local_bg.append(math.nan)

    return Mean,area,Max,Min,RawIntDen,IntDen,local_bg

def SpineBoundingBox(S,Unit,Mode,Snaps):
    """
    Rotate coordinates (x, y) by angle `theta` (radians).
    (x, y) can be arrays. 
    Returns (x_rot, y_rot).
    
    Positive theta => counterclockwise rotation in standard Cartesian coords.
    """
    cos_t = np.cos(-S.Orientation)
    sin_t = np.sin(-S.Orientation)
    
    if(Snaps == 1):
        # Apply the rotation
        x_r = cos_t * np.array(S.points)[:,0] - sin_t * np.array(S.points)[:,1]
        y_r = sin_t * np.array(S.points)[:,0] + cos_t * np.array(S.points)[:,1]
        
        xmin, xmax = x_r.min(), x_r.max()
        ymin, ymax = y_r.min(), y_r.max()
        
        S.widths.append([(xmax - xmin)*Unit,(ymax - ymin)*Unit])
    else:
        for pts in S.points:
                # Apply the rotation
                x_r = cos_t * np.array(pts)[:,0] - sin_t * np.array(pts)[:,1]
                y_r = sin_t * np.array(pts)[:,0] + cos_t * np.array(pts)[:,1]
                
                xmin, xmax = x_r.min(), x_r.max()
                ymin, ymax = y_r.min(), y_r.max()
                
                S.widths.append([(xmax - xmin)*Unit,(ymax - ymin)*Unit])
    return 0

                
def medial_axis_eval(SimVars,DendArr=None, window_instance:object=None) -> None:

    """
    function to do the full evaluation for medial axis path for the dendrite
    Args:
        Directory: Path to the data
        Mode: Mode what should be analyzeed e.g. Luminosity, Area etc.
        multichannel: for multichannel data from microscopy
        resolution: resolution of the microscopic data
        projection_type: type of the projection of the z stack
        window_instance: instance to the window where the plot stuff is shown

    Returns: None

    """
    window_instance.set_status_message.setText(window_instance.status_msg["2"])
    DendMeasure = DendriteMeasurement(SimVars= SimVars, DendArr=DendArr)

    return DendMeasure


def spine_eval(SimVars, points=np.array([]),scores=np.array([]),flags=np.array([]),clear_plot=True):

    """Evaluate and plot spine markers.

    Evaluates the spine markers based on the provided points, scores, and flags.
    Clears the plot if specified.
    Sets the status message on the GUI.
    Returns the Spine_Marker instance.

    Args:
        SimVars: The SimVars object.
        points: Array of points representing the coordinates of the spine markers. Default is an empty array.
        scores: Array of scores representing the confidence scores of the spine markers. Default is an empty array.
        flags: Array of flags representing the flags associated with the spine markers. Default is an empty array.
        clear_plot: Boolean flag indicating whether to clear the plot before plotting the spine markers. Default is True.

    Returns:
        The Spine_Marker instance representing the evaluated spine markers.
    """

    if(clear_plot):
        SimVars.frame.mpl.clear_plot()
        try:
            SimVars.frame.update_plot_handle(
                SimVars.frame.tiff_Arr[SimVars.frame.actual_timestep,SimVars.frame.actual_channel, :, :]
            )
        except:
            pass
    SimVars.frame.set_status_message.setText(SimVars.frame.status_msg["3"])
    return Spine_Marker(SimVars=SimVars, points=points,scores=scores,flags = flags)