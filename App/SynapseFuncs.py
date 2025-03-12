from .Utility import *
from skimage.feature import canny
from skimage.registration import phase_cross_correlation
from scipy.signal import medfilt2d

import json
import copy
from .Spine import Synapse
import csv
import cv2 as cv

import traceback

import roifile as rf

from .PathFinding import (
    medial_axis_path,
    downsampling_max_pool,
    curvature_polygon,
    curvature_eval,
    GetAllpointsonPath
)

from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.ndimage import distance_transform_edt

from .MPL_Widget import *
def SpineShift(tiff_Arr_small):

    """
    Input:
            tiff_Arr_small (np.array of doubles)  : Pixel values of local area surrounding a spine
    Output:
            SpineMinDir (list of ints)            : Local shifting so ROI follows spine

    Function:
           Using the phase cross correlation algorithm we can work out how to shift the ROIs
    """
    MinDir2 = np.zeros([2, tiff_Arr_small.shape[0] - 1])
    for t in range(tiff_Arr_small.shape[0] - 1):
        shift, _, _ = phase_cross_correlation(
            tiff_Arr_small[t, :, :], tiff_Arr_small[t + 1, :, :]
        )
        MinDir2[:, t] = -shift.astype(int)

    MinDirCum = np.cumsum(MinDir2, 1)
    MinDirCum = np.insert(MinDirCum, 0, 0, 1)

    return MinDirCum

def ROI_And_Neck(
    tiff_Arr_m,
    pt,
    DendArr_m,
    other_pts,
    bg,
    ErrorCorrect=False,
    sigma=1.5,
    CheckVec=[True, True, True, True],
    tol=3,
    SpineShift_flag = True,
    Mode = 'Both'):
    
    SpineMinDir = None
    neck_path = []

    if tiff_Arr_m.ndim > 2:
        tiff_Arr = tiff_Arr_m.max(axis=0)
        if(SpineShift_flag and ErrorCorrect):
            tiff_Arr_small = tiff_Arr_m[
                :,
                max(pt[1] - 70, 0) : min(pt[1] + 70, tiff_Arr_m.shape[-2]),
                max(pt[0] - 70, 0) : min(pt[0] + 70, tiff_Arr_m.shape[-1]),
            ]
            SpineMinDir = SpineShift(tiff_Arr_small).T.astype(int).tolist()
            tiff_Arr = np.array(
                [
                    np.roll(tiff_Arr_m, -np.array(m), axis=(-2, -1))[i, :, :]
                    for i, m in enumerate(SpineMinDir)
                ]
            ).max(axis=0)
    else:
        tiff_Arr = tiff_Arr_m

    Closest = 0
    if len(DendArr_m) > 1:
        Closest = [np.min(np.linalg.norm(pt - d, axis=1)) for d in DendArr_m]
        DendArr = DendArr_m[np.argmin(Closest)]
        Closest = np.argmin(Closest)
    else:
        DendArr = DendArr_m[0]
    Order0 = np.sort(
        np.argsort(np.linalg.norm(np.asarray(DendArr) - np.asarray(pt), axis=1))[
            0:2
        ]
    )

    pt_proc = np.array(projection(DendArr[Order0[0]], DendArr[Order0[1]], pt))
    OppDir = np.array(3 * pt - 2 * pt_proc).astype(int)
    OppDir[0] = np.clip(OppDir[0],10,tiff_Arr.shape[-1] - 10)
    OppDir[1] = np.clip(OppDir[1],10,tiff_Arr.shape[-2] - 10)
    dx = pt_proc[0]-pt[0]
    dy = pt_proc[1]-pt[1]
    Orientation = np.arctan2(dy,dx)
    o_arr = np.asarray(other_pts)

    neck_thresh = 0
    if(Mode=='Both'):
        xpert,DendDist = FindShape(tiff_Arr,pt,o_arr,DendArr,bg,pt_proc,sigma=sigma,tol=tol,SpineShift_flag=SpineShift_flag)
        neck_path,neck_thresh      = FindNeck(pt,pt_proc,tiff_Arr,DendArr)
    elif(Mode=='ROI'):
        xpert,DendDist = FindShape(tiff_Arr,pt,o_arr,DendArr,bg,pt_proc,sigma=sigma,tol=tol,SpineShift_flag=SpineShift_flag)
    else:
        neck_path,neck_thresh      = FindNeck(pt,pt_proc,tiff_Arr,DendArr)
        xpert = []
        DendDist = [0,0,0]

    return xpert, SpineMinDir, OppDir,Closest,DendDist,Orientation,neck_path,neck_thresh

def FindShape(
    tiff_Arr,
    pt,
    o_arr,
    DendArr,
    bg,
    pt_proc,
    ErrorCorrect=True,
    sigma=1.5,
    CheckVec=[True, True, True, True],
    tol=3,
    SpineShift_flag = True,
):

    """
    Input:
            tiff_Arr (np.array of doubles)  : Pixel values of all the tiff files
            pt       ([int,int])            : Point of interest for roi
            DendArr_m (np.array of doubles) : Location of the dendritic branch
            other_pts (np.array of ints)    : Locations of other rois
            bg (double)                     : value of background
            ErrorCorrect (bool)             : Flag to see if perturbations of pt
                                              should also be analysed
            sigma (double)                  : Value of canny image variance
            CheckVec (list of bools)        : Which of the conditions we impose
            tol (int)                       : How many strikes we accept before stopping
    Output:
            xprt (np.array of ints)         : shape of roi
            SpineMinDir (list of ints)      : Local shifting so ROI follows spine
            OppDir (np.array of ints)       : Vector pointing away from dendrite

    Function:
            Using a set of rules based on the luminosity we aim to encircle the spine
            and pass this out via xpert. Current rules are, "Sanity check", "Fall-off"
            "Dendrite criterion", "Overlap criterion" and "Fallback criterion"
    """

    DendDist = None

    cArr = canny(tiff_Arr, sigma=sigma)

    strikes = 0
    Directions = {
        "N": [0, 1, True, "S", 0, strikes, True],
        "NW": [-1, 1, True, "SE", 0, strikes, True],
        "W": [-1, 0, True, "E", 0, strikes, True],
        "SW": [-1, -1, True, "NE", 0, strikes, True],
        "S": [0, -1, True, "N", 0, strikes, True],
        "SE": [1, -1, True, "NW", 0, strikes, True],
        "E": [1, 0, True, "W", 0, strikes, True],
        "NE": [1, 1, True, "SW", 0, strikes, True],
    }

    xpert = np.array([pt, pt, pt, pt, pt, pt, pt, pt])
    maxval = tiff_Arr[pt[1], pt[0]]

    if CheckVec[3]:
        for keys in Directions.keys():
            for x, y in zip(DendArr[:-1, :], DendArr[1:, :]):
                lam, mu = crosslen(x, y, pt, pt + Directions[keys][:2])
                if (mu < 0) or (lam > 1 or lam < 0):
                    Directions[keys][-1] = True
                else:
                    Directions[keys][-1] = False
                    break

    while any([x[2] for x in Directions.values()]):
        for j, keys in enumerate(Directions.keys()):
            if Directions[keys][2]:
                maxval = max(maxval, tiff_Arr[xpert[j][1], xpert[j][0]])
                xpert[j] = xpert[j] + Directions[keys][:2]
                Directions[keys][4] = np.linalg.norm(pt - xpert[j])

                # Sanity check
                if (
                    xpert[j] > [tiff_Arr.shape[1] - 2, tiff_Arr.shape[0] - 2]
                ).any() or (xpert[j] < 1).any():
                    Directions[keys][-2] += 3
                    Directions[keys][2] = False
                    break

                # Contour check
                if cArr[xpert[j][1], xpert[j][0]] == True and CheckVec[0]:
                    if Directions[keys][4] <= 4:
                        if Directions[keys][4] > 1:
                            Directions[keys][-2] += 1
                    elif Directions[keys][4] <= 8:
                        Directions[keys][-2] += 2
                    else:
                        Directions[keys][-2] += 3

                # Fall off criterion
                if (
                    tiff_Arr[xpert[j][1], xpert[j][0]] < 4 * bg
                    or 3 * tiff_Arr[xpert[j][1], xpert[j][0]] < maxval
                ) and CheckVec[1]:
                    if Directions[keys][4] <= 4:
                        if Directions[keys][4] > 1:
                            Directions[keys][-2] += 1
                    elif Directions[keys][4] <= 8:
                        Directions[keys][-2] += 2
                    else:
                        Directions[keys][-2] += 3

                # Dendrite criterion
                if CheckVec[2]:
                    if (
                        np.linalg.norm(pt - xpert[j])
                        > np.linalg.norm(pt_proc - xpert[j])
                        and np.linalg.norm(pt_proc - pt) > 5
                    ):
                        Directions[keys][-2] += 3

                # Increasing criterion
                if (
                    not Directions[keys][-1]
                    and Directions[keys][4] > 5
                    and tiff_Arr[xpert[j][1], xpert[j][0]] > 1.5 * maxval
                ) and CheckVec[3]:
                    Directions[keys][-2] += 1

                # Overlap criterion
                if not o_arr.size == 0:
                    if np.any(
                        np.linalg.norm(xpert[j] - o_arr, axis=1)
                        < np.linalg.norm(xpert[j] - pt)
                    ):
                        Directions[keys][-2] += 3

                # Fallback criterion
                if (
                    not Directions[Directions[keys][3]][2]
                    and Directions[keys][4] > 2 * Directions[Directions[keys][3]][4]
                ):
                    Directions[keys][-2] += 1

                if Directions[keys][-2] >= tol:
                    Directions[keys][2] = False

    if ErrorCorrect:
        o_arr2 = np.delete(o_arr, np.where(np.all(o_arr == pt, axis=1)), axis=0)
        xpert1, _ = FindShape(
            tiff_Arr, pt + [0, 1], o_arr2, DendArr,bg,pt_proc, False, sigma, CheckVec, tol
        )
        xpert2, _ = FindShape(
            tiff_Arr, pt + [0, -1], o_arr2, DendArr,bg,pt_proc, False, sigma, CheckVec, tol
        )
        xpert3, _ = FindShape(
            tiff_Arr, pt + [1, 0], o_arr2, DendArr,bg,pt_proc, False, sigma, CheckVec, tol
        )
        xpert4, _ = FindShape(
            tiff_Arr, pt + [-1, 0], o_arr2, DendArr,bg,pt_proc, False, sigma, CheckVec, tol
        )
        xpert = np.stack((xpert, xpert1, xpert2, xpert3, xpert4)).mean(0)
        dists = np.linalg.norm(np.array(xpert)-pt_proc,axis=1)
        DendDist = np.array([dists.max(),np.linalg.norm(pt_proc - pt),dists.min()])
        xpert = xpert.tolist()

    return xpert,DendDist


def SynDistance(SynArr, DendArr_m, Unit):

    """
    Input:
            SynArr  (list of synapses)
            DendArr_m (list of np.array of doubles) : Location of the dendritic branches
            Unit                                    : The size of each pixel
            Mode (String)                           : Type of data we are collecting
    Output:
            SynArr  (list of synapses)

    Function:
            Find the distance to the start of the dendrite
    """

    # Horizontal distance
    for S in SynArr:
        DendArr = DendArr_m[S.closest_Dend]
        S.distance = SynDendDistance(S.location, DendArr, DendArr[0]) * Unit

    return SynArr

def FindNeck(SpineC,DendProj,image,DendArr):


    start = SpineC.astype(int)[::-1]
    end = DendProj.astype(int)[::-1]

    bbox_min = np.max([np.min(np.stack([end,start]),axis=0)-50,[0,0]],axis=0)
    bbox_max = np.min([np.max(np.stack([end,start]),axis=0)+50,image.shape],axis=0)

    img = image[bbox_min[0]:bbox_max[0]+1,bbox_min[1]:bbox_max[1]+1]
    median = medfilt2d(img, kernel_size=5)
    success = False
    k = 1
    while success == False:
        try:
            median_thresh = median >= 2*np.mean(median)//(1.1**(k-1))

            medial_axis, length = medial_axis_path(
                mesh=median_thresh, start=start-bbox_min, end=end-bbox_min)
            success = True
            neck_thresh = 2*np.mean(median)//(1.1**(k-1))
        except:
            k+=1
            if(k==50):
                median_thresh = median >= 0

                medial_axis, length = medial_axis_path(
                    mesh=median_thresh, start=start-bbox_min, end=end-bbox_min)
                success = True
                neck_thresh = 0

    x, y = medial_axis[:, 0], medial_axis[:, 1]
    if((x == x[0]).all() or (y == y[0]).all()):
        curvature_sampled = np.array([x[[0,len(x)//3,2*len(x)//3,-1]].T, y[[0,len(x)//3,2*len(x)//3,-1]].T]).T
    else:
        Tx, Ty, Hx, Hy, T, H = curvature_polygon(x, y)
        H = H / len(H)
        sampling, _, _ = curvature_dependent_sampling(H, 100)
        x, y = x[sampling], y[sampling]
        curvature_sampled = np.array([x.T, y.T]).T

    shifted_path = curvature_sampled+bbox_min[::-1]

    distances = np.array([np.min(np.linalg.norm(t-DendArr,axis=-1)) for t in shifted_path])
    cutoff = np.argmin(distances)

    return shifted_path[:cutoff].tolist(),neck_thresh.astype(np.float64)

def FindNeckWidth(neck_path,image, thresh, max_neighbours: int = 1, sigma: int = 10, width_factor: int=0.1):

    all_points = GetAllpointsonPath(np.round(neck_path).astype(int))

    gaussian_x = gaussian_filter1d(        
        input=all_points[:, 1], mode="nearest", sigma=sigma    
    ).astype(int)
    gaussian_y = gaussian_filter1d(
        input=all_points[:, 0], mode="nearest", sigma=sigma
    ).astype(int)
    smoothed_all_pts = np.stack((gaussian_y, gaussian_x), axis=1)

    median = medfilt2d(image, kernel_size=5)
    if(thresh > 0):
        median_thresh = median >= thresh
    else:
        median_thresh = median >= np.mean(median)
    width_arr, degrees = getWidthnew(
        median_thresh,
        smoothed_all_pts,
        sigma=sigma,
        max_neighbours=max_neighbours,
        width_factor=width_factor
    )
    mask = np.zeros(shape=image.shape)

    for pdx, p in enumerate(smoothed_all_pts[1:]):
        rr, cc = ellipse(
            p[1],
            p[0],
            width_arr[pdx],
            0.1,
            rotation=degrees[pdx],
            shape=image.shape,
        )
        mask[rr, cc] = 1
    
    gaussian_mask = (gaussian_filter(input=mask, sigma=sigma) >= np.mean(mask)).astype(np.uint8)
    if(gaussian_mask.sum() == gaussian_mask.size):
        return 0,0
    contours, _ = cv.findContours(gaussian_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Compute the distance transform
    dist_transform = distance_transform_edt(gaussian_mask)

    # For each point along your centerline (smoothed_all_pts), sample the distance
    widths = []
    for p in smoothed_all_pts:
        y, x = p  # note: p is (y, x)
        # Make sure indices are within bounds
        if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]:
            # The width at this point is approximately twice the distance to the edge
            widths.append(2 * dist_transform[y, x])

    # Now you can compute the median or mean width

    final_width = np.mean(widths) if widths else None
    

    return contours,final_width

def SynDendDistance(loc, DendArr, loc0):

    """
    Input:
            loc       ([int,int])         : Point of interest of spine
            DendArr (np.array of doubles) : Location of the dendritic branch
            loc0       ([int,int])         : Point of interest of closest stim
    Output:
            Distance (real)

    Function:
            Find the distance of spine along the dendrite
    """

    DoneDist = [np.linalg.norm(d1 - d2) for d1, d2 in zip(DendArr[:-1], DendArr[1:])]
    Order0 = np.sort(
        np.argsort(np.linalg.norm(np.asarray(DendArr) - np.asarray(loc0), axis=1))[0:2]
    )
    S0Proc = projection(DendArr[Order0[0]], DendArr[Order0[1]], loc0)
    S0Dist = np.linalg.norm(np.asarray(DendArr)[Order0] - S0Proc, axis=1)

    Order = np.sort(
        np.argsort(np.linalg.norm(np.asarray(DendArr) - np.asarray(loc), axis=1))[0:2]
    )
    SProc = projection(DendArr[Order[0]], DendArr[Order[1]], loc)
    Distance = 0
    if ((Order0 == Order)).all():
        Distance = np.linalg.norm(np.array(SProc) - np.array(S0Proc))
    elif Order0[0] >= Order[1]:
        Distance = np.linalg.norm(SProc - DendArr[Order[1]])
        for i in range(Order[1], Order0[0]):
            Distance += DoneDist[i]
        Distance += S0Dist[0]
    elif Order0[1] <= Order[0]:
        Distance = np.linalg.norm(SProc - DendArr[Order[0]])
        for i in range(Order[0], Order0[1], -1):
            Distance += DoneDist[i - 1]
        Distance += S0Dist[1]
    else:
        Distance = np.linalg.norm(np.array(SProc) - np.array(S0Proc))

    return Distance

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
def to_list(val):
    return val.tolist() if hasattr(val, 'tolist') else val

def SaveSynDict(SynArr, Dir, Mode,xLims):

    """
    Input:
            SynArr  (list of synapses)
            bg (list of doubles)         : Background values of the snapshots
            Dir (String)                 : Super directory we are looking at
            Mode (String)                : Type of data we are collecting
    Output:
            N/A

    Function:
            Save list of spines as json file
    """
    modifiedSynArr = copy.deepcopy(SynArr)
    for S in modifiedSynArr:
        if hasattr(S, 'neck_contours'):
            delattr(S, 'neck_contours')
    if(len(xLims[0])==0):
        Lims = np.array([0,0])
    else:
        Lims = np.array([xLims[0][0],xLims[1][0]])
   
    # For each object, subtract Lims where possible and convert attributes to lists.
    for S in modifiedSynArr:
        # Subtract Lims from attributes if they exist.
        for attr in ['points', 'location', 'bgloc']:
            if hasattr(S, attr):
                try:
                    # Subtraction should work if the attribute is a NumPy array or similar.
                    setattr(S, attr, getattr(S, attr) - Lims)
                except Exception as e:
                    pass
        
        # Convert attributes to lists.
        # For 'points', we handle nested conversion if needed.
        if hasattr(S, 'points'):
            try:
                if isinstance(S.points, list):
                    S.points = [to_list(item) for item in S.points]
                else:
                    S.points = to_list(S.points)
            except Exception as e:
                pass
        
        for attr in ['location', 'distance_to_Dend', 'bgloc', 'neck', 'neck_mean']:
            if hasattr(S, attr):
                try:
                    setattr(S, attr, to_list(getattr(S, attr)))
                except Exception as e:
                    pass

    if Mode == "Area":
        with open(Dir + "Synapse_a.json", "w") as fp:
            json.dump([vars(S) for S in modifiedSynArr], fp, indent=4,cls=NumpyEncoder)
    elif Mode == "Luminosity":
        with open(Dir + "Synapse_l.json", "w") as fp:
            json.dump([vars(S) for S in modifiedSynArr], fp, indent=4,cls=NumpyEncoder)
    else:
        with open(Dir + "Synapse.json", "w") as fp:
            json.dump([vars(S) for S in modifiedSynArr], fp, indent=4,cls=NumpyEncoder)

    return 0


def ReadSynDict(Dir, SimVars):

    """
    Input:
            Dir (String)   : Super directory we are looking at
            nSnaps (int)   : Number of snapshots
            Unit (double)  : The size of each pixel
            Mode (String)  : Type of data we are collecting
    Output:
            SynArr  (list of synapses)
    Function:
            Read Json file to obtain saved list of synapses
    """
    unit   = SimVars.Unit
    Mode   = SimVars.Mode
    nSnaps = SimVars.Snapshots
    NecessaryKeys = ['location','bgloc','type','distance','points','shift','channel','local_bg','closest_Dend','distance_to_Dend','Orientation','neck', 'neck_thresh']

    AppliedVals = {'points':[],'dist':None,'type':None,'shift':[],'channel':0,'local_bg':0,'closest_Dend':0,'DendDist' : [0,0,0],'Orientation' : 0,'neck' : [], 'neck_thresh' : 0}

    if(len(SimVars.xLims)==0):
        xLim = 0
        yLim = 0
    else:
        xLim   = SimVars.xLims[0]
        yLim   = SimVars.yLims[0]

    if(Mode=="Area"):
        FileName="Synapse_a.json"
        FileName2="Synapse_l.json"
    else:
        FileName="Synapse_l.json"
        FileName2="Synapse_a.json"

    # Should have a check if the file exists to see if it can be read
    try:
        try:
            with open(Dir + FileName, "r") as fp:
                temp = json.load(fp)
        except:
            with open(Dir + FileName2, "r") as fp:
                temp = json.load(fp)
        SynArr = []

        for t in temp:

            for k in NecessaryKeys:
                if k in t:
                    AppliedVals[k] = t[k]

            try:
                SynArr.append(
                    Synapse(
                        (AppliedVals["location"]+np.array([yLim,xLim])).tolist(),
                        (AppliedVals["bgloc"]+np.array([yLim,xLim])).tolist(),
                        Syntype=AppliedVals["type"],
                        dist=AppliedVals["distance"],
                        pts=(AppliedVals["points"]+np.array([yLim,xLim])).tolist(),
                        shift=AppliedVals["shift"],
                        channel=AppliedVals["channel"],
                        local_bg=AppliedVals["local_bg"],
                        closest_Dend=AppliedVals["closest_Dend"],
                        DendDist = AppliedVals["distance_to_Dend"],
                        Orientation = AppliedVals["Orientation"],
                        neck       = AppliedVals["neck"],
                        neck_thresh       = AppliedVals["neck_thresh"]

                    )
                )
            except:
                pts = []
                for pt in AppliedVals['points']: 
                    pts.append((pt+np.array([yLim,xLim])).tolist())
                SynArr.append(
                    Synapse(
                        (AppliedVals["location"]+np.array([yLim,xLim])).tolist(),
                        AppliedVals["bgloc"],
                        Syntype=AppliedVals["type"],
                        dist=AppliedVals["distance"],
                        pts=pts,
                        shift=AppliedVals["shift"],
                        channel=AppliedVals["channel"],
                        local_bg=AppliedVals["local_bg"],
                        closest_Dend=AppliedVals["closest_Dend"],
                        DendDist = AppliedVals["distance_to_Dend"],
                        Orientation = AppliedVals["Orientation"],
                        neck       = AppliedVals["neck"],
                        neck_thresh       = AppliedVals["neck_thresh"]
                    )
                )
                SynArr[-1].shift = np.zeros([nSnaps, 2]).tolist()
    except Exception as e:
       print(e)
       return []
    return SynArr

def SpineSave_csv(Dir,Spine_Arr,nChans,nSnaps,Mode,xLims,local_shift):
    """
    Save Spine data into CSV files.

    Args:
        Dir (str): Directory path where the CSV files will be saved.
        Spine_Arr (list): List of synapse objects.
        nChans (int): Number of channels.
        nSnaps (int): Number of snapshots.
        Mode (str): Mode for saving the data. Can be 'Luminosity' or 'Area'.
        xLims (list): List containing limits of the tiff_Arr.
    Returns:
        None
    """   
    OnlySoma = False
    if(np.all([S.type == 2 for S in Spine_Arr])):
        OnlySoma = True
    else:
        FirstNonSoma = np.argmin([S.type == 2 for S in Spine_Arr])

    if(len(xLims[0])==0):
        Lims = np.array(0)
    else:
        Lims = np.array([xLims[0][0],xLims[1][0]])
    if(Mode=='Luminosity'):
        custom_header =(['', 'type','location','bgloc','area','distance','closest_Dend','Max, dist to Dend',
            'Center dist to dend','Min, dist to Dend','Widths'] + 
        ['Timestep '+ str(i) +' (mean)' for i in range(1,nSnaps+1)] +
        ['Timestep '+ str(i) +' (min)' for i in range(1,nSnaps+1)] +
        ['Timestep '+ str(i) +' (max)' for i in range(1,nSnaps+1)] +
        ['Timestep '+ str(i) +' (RawIntDen)' for i in range(1,nSnaps+1)] +
        ['Timestep '+ str(i) +' (IntDen)' for i in range(1,nSnaps+1)] +
        ['Timestep '+ str(i) +' (bg mean)' for i in range(1,nSnaps+1)])
        if(not OnlySoma):
            if(local_shift):
                custom_header +=['Timestep '+ str(i+1) +' (neck length)' for i in range(1,nSnaps+1)] + ['Timestep '+ str(i+1) +' (neck width)' for i in range(1,nSnaps+1)] + ['Timestep '+ str(i) +' (neck mean)' for i in range(1,nSnaps+1)]
            else:
                custom_header +=['Neck length'] + ['Neck width'] + ['Timestep '+ str(i) +' (neck mean)' for i in range(1,nSnaps+1)]

        for c in range(nChans):
            csv_file_path = Dir+'Synapse_l_Channel_' + str(c)+'.csv'
            with open(csv_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(custom_header) 
                if(OnlySoma):
                    for i,s in enumerate(Spine_Arr):
                        row = ['Spine: '+str(i),s.type,
                                         str(s.location-Lims),str(s.bgloc-Lims),s.area,s.distance,s.closest_Dend
                                         ]+[str(d) for d in s.distance_to_Dend] + [str(s.widths)]
                        

                        row.extend(s.mean[c])
                        row.extend(s.min[c])
                        row.extend(s.max[c])
                        row.extend(s.RawIntDen[c])
                        row.extend(s.IntDen[c])
                        row.extend(s.local_bg[c])
                        writer.writerow(row)
                else:
                    for i,s in enumerate(Spine_Arr):
                        if(s.type == 2):
                            row = ['Spine: '+str(i),s.type,
                                             str(s.location-Lims),str(s.bgloc-Lims),s.area,s.distance,s.closest_Dend
                                             ]+[str(d) for d in s.distance_to_Dend] + [str(s.widths)]
                            

                            row.extend(s.mean[c])
                            row.extend(s.min[c])
                            row.extend(s.max[c])
                            row.extend(s.RawIntDen[c])
                            row.extend(s.IntDen[c])
                            row.extend(s.local_bg[c])
                            if(local_shift):
                                row += ['' for n in range(nSnaps)] +['' for n in range(nSnaps)]
                            else:
                                row += [''] +['']
                            row.extend(np.zeros(nSnaps))
                            writer.writerow(row)
                        else:
                            row = ['Spine: '+str(i),s.type,

                                         str(s.location-Lims),str(s.bgloc-Lims),s.area,s.distance,s.closest_Dend
                                         ]+[str(d) for d in s.distance_to_Dend] + [str(s.widths)]
                        

                            row.extend(s.mean[c])
                            row.extend(s.min[c])
                            row.extend(s.max[c])
                            row.extend(s.RawIntDen[c])
                            row.extend(s.IntDen[c])
                            row.extend(s.local_bg[c])
                            if(len(s.neck_mean)==0):
                                row.extend(np.zeros(nSnaps))
                                row += ['' for n in s.neck_length] +['' for n in s.neck_width]
                            else:
                                row += [str(n) for n in s.neck_length] +[str(n) for n in s.neck_width]
                                row.extend(s.neck_mean[c])
                            writer.writerow(row)

    else: 
        custom_header =(['', 'type','location','distance','closest_Dend','Max. dist to Dend',
            'Center dist to dend','Min. dist to Dend'] + 
        ['Timestep '+ str(i) +' (area)' for i in range(1,nSnaps+1)] +  
        ['Timestep '+ str(i) +' (mean)' for i in range(1,nSnaps+1)] +
        ['Timestep '+ str(i) +' (min)' for i in range(1,nSnaps+1)] +
        ['Timestep '+ str(i) +' (max)' for i in range(1,nSnaps+1)] +
        ['Timestep '+ str(i) +' (RawIntDen)' for i in range(1,nSnaps+1)] +
        ['Timestep '+ str(i) +' (IntDen)' for i in range(1,nSnaps+1)] + 
        ['Timestep '+ str(i) +' (Widths)' for i in range(1,nSnaps+1)])
        if(not OnlySoma):
            custom_header +=['Timestep '+ str(i+1) +' (neck length)' for i in range(1,nSnaps+1)] + ['Timestep '+ str(i+1) +' (neck width)' for i in range(1,nSnaps+1)] + ['Timestep '+ str(i) +' (neck mean)' for i in range(1,nSnaps+1)]

        for c in range(nChans):
            csv_file_path = Dir+'Synapse_a_Channel_' + str(c)+'.csv'
            with open(csv_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(custom_header) 
                if(OnlySoma):
                    for i,s in enumerate(Spine_Arr):
                        row = ['Spine: '+str(i),s.type,
                                         str(s.location-Lims),s.distance,s.closest_Dend
                                         ]+[str(d) for d in s.distance_to_Dend]
                        row.extend(s.area[c])
                        row.extend(s.mean[c])
                        row.extend(s.min[c])
                        row.extend(s.max[c])
                        row.extend(s.RawIntDen[c])
                        row.extend(s.IntDen[c])
                        row.extend([str(w) for w in s.widths])
                        writer.writerow(row)   
                else:
                    for i,s in enumerate(Spine_Arr):
                        if(s.type == 2):
                            row = ['Spine: '+str(i),s.type,
                                             str(s.location-Lims),s.distance,s.closest_Dend
                                             ]+[str(d) for d in s.distance_to_Dend]
                            
                            row.extend(s.area[c])
                            row.extend(s.mean[c])
                            row.extend(s.min[c])
                            row.extend(s.max[c])
                            row.extend(s.RawIntDen[c])
                            row.extend(s.IntDen[c])
                            row.extend([str(w) for w in s.widths])
                            row += ['' for n in range(nSnaps)] +['' for n in range(nSnaps)]
                            row.extend(np.zeros(nSnaps))
                            writer.writerow(row)
                        else:
                            row = ['Spine: '+str(i),s.type,
                                         str(s.location-Lims),s.distance,s.closest_Dend
                                         ]+[str(d) for d in s.distance_to_Dend]
                            row.extend(s.area[c])
                            row.extend(s.mean[c])
                            row.extend(s.min[c])
                            row.extend(s.max[c])
                            row.extend(s.RawIntDen[c])
                            row.extend(s.IntDen[c])
                            row.extend([str(w) for w in s.widths])
                            if(len(s.neck_mean)==0):
                                row += [str(n) for n in s.neck_length] +['' for n in s.neck_width]
                                row.extend(np.zeros(nSnaps))
                            else:
                                row += [str(n) for n in s.neck_length] +[str(n) for n in s.neck_width]
                                row.extend(s.neck_mean[c])
                            writer.writerow(row)

def SpineSave_imj(Dir,Spine_Arr):
    os.mkdir(path=Dir+'ImageJ/')
    Dir2 = Dir+'ImageJ/'
    for i,S in enumerate(Spine_Arr):
        pts = S.points
        if(len(pts[0])>2):
            for j,p in enumerate(pts):
                roi = rf.ImagejRoi.frompoints(p)
                roi.roitype = rf.ROI_TYPE.POLYGON
                roi.tofile(Dir2+'Spine_'+str(i)+'_t'+str(j)+'.roi')
        else:
            roi = rf.ImagejRoi.frompoints(pts)
            roi.roitype = rf.ROI_TYPE.POLYGON
            roi.tofile(Dir2+'Spine_'+str(i)+'.roi')
        if(S.type < 2):
            try:
                necks = S.neck_contours
                if(len(necks[0])>2):
                    for j,p in enumerate(necks):
                        roi = rf.ImagejRoi.frompoints(p)
                        roi.roitype = rf.ROI_TYPE.POLYGON
                        roi.tofile(Dir2+'Spine_neck_'+str(i)+'_t'+str(j)+'.roi')
                else:
                    roi = rf.ImagejRoi.frompoints(necks)
                    roi.roitype = rf.ROI_TYPE.POLYGON
                    roi.tofile(Dir2+'Spine_neck_'+str(i)+'.roi')
            except:
                pass