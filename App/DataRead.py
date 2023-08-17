from skimage.registration import phase_cross_correlation
from skimage.draw import polygon
import tifffile as tf
import math
import re

from .Utility import *
from .Spine    import *
from .Dendrite import *

from .SynapseFuncs import *

import json

from PyQt5.QtCore import QCoreApplication


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

    md = getMetadata(Dir + "/" + File_Names[0])

    if File_Names[0].endswith(".lsm"):
        scale = getScale(Dir + "/" + File_Names[0])
    else:
        scale = scale

    tiff_Arr = []
    for i, x in enumerate(File_Names):
        md = getMetadata(Dir + "/" + x)
        temp = tf.imread(Dir + x)
        temp_mod = temp.reshape(md[1:])
        if not Channels:
            temp_mod = z_type(temp_mod, axis=1, keepdims=True)
        tiff_Arr.append(z_type(temp_mod, axis=0))

    md[0] = len(tiff_Arr)
    if not z_type == None:
        md[1] = 1
    md[2:] = tiff_Arr[0].shape

    return np.array(tiff_Arr), Times, md, scale


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
        return getTifDimenstions(filename)
    elif filename.endswith(".lsm"):
        return getLSMDimensions(filename)
    else:
        if frame is None:
            print("Unsupported file format found. contact admin")
        # TODO: Format print as pop-up/In the main window
        exit()


def getScale(filename):
    tf_file = tf.TiffFile(filename)
    if filename.endswith(".tif"):
        return 0.114
    elif filename.endswith(".lsm"):
        return tf_file.lsm_metadata["ScanInformation"]["SampleSpacing"]
    else:
        print("Unsupported file format found. contact admin")

def getTifDimenstions(filename):

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
        if ".lsm" in x or ".tif" in x:
            File_Names.append(x)

    regex = re.compile(".\d+")
    File_Names_int = [re.findall(regex, f)[0] for f in File_Names]

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

    tf,SimVars.xLims,SimVars.yLims = ShiftArr(tiff_Arr, MinDirCum)
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
        if(SimVars.Mode=="Luminosity" or Snaps==1):
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
        else:
            for S in SynArr:
                frame.set_status_message.setText(frame.set_status_message.text()+'.')
                Mean,Area,Max,Min,RawIntDen,IntDen = MeasureShape(S, tiff_Arr[:,i,:,:], SimVars,Snaps)
                S.max.append(Max)
                S.min.append(Mean)
                S.RawIntDen.append(RawIntDen)
                S.IntDen.append(IntDen)
                S.mean.append(Mean)
                S.area.append(Area)
    else:
        if(SimVars.Mode=="Luminosity" or Snaps==1):
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
        else:
            for S in SynArr:
                frame.set_status_message.setText(frame.set_status_message.text()+'.')
                Mean,Area,Max,Min,RawIntDen,IntDen = MeasureShape(S, tiff_Arr[:,SimVars.frame.actual_channel,:,:], SimVars,Snaps)
                S.max.append(Max)
                S.min.append(Mean)
                S.RawIntDen.append(RawIntDen)
                S.IntDen.append(IntDen)
                S.mean.append(Mean)
                S.area.append(Area)
    return 0

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

                
def medial_axis_eval(SimVars,tiff_Arr,DendArr=None, window_instance:object=None) -> None:

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
    DendMeasure = DendriteMeasurement(SimVars= SimVars, tiff_Arr=tiff_Arr,DendArr=DendArr)

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