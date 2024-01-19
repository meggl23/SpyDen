from .LineInteractor import *
from .MPL_Widget import *

from scipy.signal import medfilt2d

from .PathFinding import (
    medial_axis_path,
    downsampling_max_pool,
    curvature_polygon,
    curvature_eval,
    GetAllpointsonPath
)

from matplotlib.patches import Polygon

from scipy.ndimage import gaussian_filter1d, gaussian_filter
import cv2 as cv
from skimage.draw import ellipse
import csv
import math
import json

import roifile as rf

class Dendrite:
    """this class holds all the params for the dendrite like tif_arr"""

    def __init__(self, tiffstack, SimVars):
        self.tiff_arr = tiffstack
        self.Morphologie = tiffstack[0,0,:,:]
        self.median_filtered = medfilt2d(
                                            input=self.Morphologie, kernel_size=5
                                            )
        self.SimVars = SimVars
        self.len_y = len(self.Morphologie[:, 0])
        self.len_x = len(self.Morphologie[0, :])
        self.medial_axis_path_downsampled = None
        self.medial_axis_path = None
        self.thresh = False
        self.mean = np.mean(self.Morphologie)
        self.median_thresh = self.median_filtered >= self.mean
        self.length = None
        self.control_points = None
        self.dendritic_surface_matrix = None
        self.contours = None
        self.actual_channel = self.SimVars.frame.actual_channel
        self.actual_timestep = self.SimVars.frame.actual_timestep
        self.WidthFactor = None


    def UpdateParams(self,tiffstack):
        self.tiff_arr = tiffstack
        self.Morphologie = tiffstack[0,0,:,:]
        self.median_filtered = medfilt2d(
                                            input=self.Morphologie, kernel_size=5
                                            )
        self.len_y = len(self.Morphologie[:, 0])
        self.len_x = len(self.Morphologie[0, :])
        self.mean = np.mean(self.Morphologie)
        self.median_thresh = self.median_filtered >= self.mean

    def calc_medial_axis_path(self) -> None:
        """
        calculates the control points of the medial axis path
        medial axis path due to curvature sampling
        Returns: None

        """
        actual_image = self.tiff_arr[self.actual_timestep, self.actual_channel, :, :]
        median = medfilt2d(actual_image, kernel_size=5)
        if self.thresh is not False:

            self.median_thresh = median >= self.thresh
        else:
            self.median_thresh = median >= np.mean(median)

        if self.len_y <= 512 and self.len_x <= 512:

            start = (self.coords[0] ).astype(int)
            end = (self.coords[1] ).astype(int)
            downsampled_medial_axis, length = medial_axis_path(
                mesh=self.median_thresh, start=start, end=end, scale=self.SimVars.Unit
            )
            self.downsampled_medial_axis = downsampled_medial_axis
            self.length = length
            self.medial_axis = self.downsampled_medial_axis
            x, y = self.medial_axis[:, 0], self.medial_axis[:, 1]
            Tx, Ty, Hx, Hy, T, H = curvature_polygon(x, y)
            H = H / len(H)
            sampling, boarders, aver = curvature_dependent_sampling(H, 25)
            x, y = x[sampling], y[sampling]
            self.curvature_sampled = np.array([x.T, y.T]).T

        else:
            downsampled = downsampling_max_pool(
                img=self.median_thresh, kernel_size=4, stride=2
            )
            start = (self.coords[0]/2).astype(int)
            end = (self.coords[1]/2).astype(int)
            downsampled_medial_axis, length = medial_axis_path(
                mesh=downsampled, start=start, end=end, scale=self.SimVars.Unit
            )
            self.downsampled_medial_axis = downsampled_medial_axis
            self.length = length
            self.medial_axis = self.downsampled_medial_axis*2

            x, y = self.medial_axis[:, 0], self.medial_axis[:, 1]
            Tx, Ty, Hx, Hy, T, H = curvature_polygon(x, y)
            H = H / len(H)
            sampling, boarders, aver = curvature_dependent_sampling(H, 25)
            x, y = x[sampling], y[sampling]
            self.curvature_sampled = np.array([x.T, y.T]).T




    def get_control_points(self) -> np.ndarray:
        """

        Returns:control points of medial axis path

        """
        self.control_points = self.lineinteract.getPolyXYs()
        return self.control_points

    def set_surface_contours(self, max_neighbours: int = 6, sigma: int = 10, width_factor: int=1) -> None:
        """

        Args:
            max_neighbours: number of the maximum increase (Pixels) of width of the next step
            sigma:

        Returns:

        """
        all_points = GetAllpointsonPath(self.control_points)[:, :]

        gaussian_x = gaussian_filter1d(
            input=all_points[:, 1], mode="nearest", sigma=sigma
        ).astype(int)
        gaussian_y = gaussian_filter1d(
            input=all_points[:, 0], mode="nearest", sigma=sigma
        ).astype(int)
        self.smoothed_all_pts = np.stack((gaussian_y, gaussian_x), axis=1)
        actual_image = self.tiff_arr[self.actual_timestep, self.actual_channel, :, :]
        median = medfilt2d(actual_image, kernel_size=5)
        if self.thresh is not False:
            median_thresh = median >= self.thresh

        else:
            median_thresh = median >= np.mean(median)
        width_arr, degrees = getWidthnew(
            median_thresh,
            self.smoothed_all_pts,
            sigma=sigma,
            max_neighbours=max_neighbours,
            width_factor=width_factor
        )
        mask = np.zeros(shape=self.Morphologie.shape)

        self.dend_stat = np.zeros(shape= (len(self.smoothed_all_pts), 5))
        if(self.SimVars.multitime_flag):
            Snaps = self.SimVars.Snapshots
        else:
            Snaps = 1
        if(self.SimVars.multiwindow_flag):
            Chans = self.SimVars.Channels
        else:
            Chans = 1
        self.dend_lumin = np.zeros(shape= (len(self.smoothed_all_pts), Snaps,Chans))
        self.dend_lumin_ell = np.zeros(shape= (len(self.smoothed_all_pts), Snaps,Chans))
        for pdx, p in enumerate(self.smoothed_all_pts):
            self.dend_stat[pdx, 0] = p[1]
            self.dend_stat[pdx, 1] = p[0]
            self.dend_stat[pdx, 2] = width_arr[pdx]/width_factor
            self.dend_stat[pdx, 3] = 2
            self.dend_stat[pdx, 4] = degrees[pdx]

            mask = self.GenEllipse(mask,p,pdx,degrees,width_arr,Snaps,Chans)

        gaussian_mask = (gaussian_filter(input=mask, sigma=sigma) >= np.mean(mask)).astype(np.uint8)
        self.contours, _ = cv.findContours(gaussian_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        self.dendritic_surface_matrix= gaussian_mask

    def GenEllipse(self,mask,p,pdx,degrees,width_arr,Snaps,Chans):
        rr, cc = ellipse(
            p[1],
            p[0],
            width_arr[pdx],
            2,
            rotation=degrees[pdx],
            shape=self.Morphologie.shape,
        )
        mask[rr, cc] = 1
        for i in range(Snaps):
            for j in range(Chans):
                self.dend_lumin[pdx,i,j] = self.tiff_arr[i,j,p[1],p[0]]
                self.dend_lumin_ell[pdx,i,j] = (self.tiff_arr[i,j]*mask).sum()/np.sum(mask)

        return mask

    def get_contours(self) -> tuple:
        return self.contours

    def get_dendritic_surface_matrix(self) -> np.ndarray:
        return self.dendritic_surface_matrix


class DendriteMeasurement:
    """this class handles the communication between the gui and the data calculation classes
    this class is furthermore used to add functionality to the gui classes"""

    def __init__(self, SimVars, DendArr=None):
        self.coords = []
        self.SimVars = SimVars
        self.click_conn = self.SimVars.frame.mpl.canvas.mpl_connect("button_press_event", self.on_left_click)
        self.press_conn = self.SimVars.frame.mpl.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.AnotherDendFlag = True
        self.thresh = False
        if(DendArr is not None):
            for Dend in DendArr:
                Dend.pol = Polygon(
                Dend.curvature_sampled, fill=False, closed=False, animated=True
                )
                self.SimVars.frame.mpl.axes.add_patch(Dend.pol)
                Dend.lineinteract = LineInteractor(
                    self.SimVars.frame.mpl.axes, self.SimVars.frame.mpl.canvas, Dend.pol,not self.AnotherDendFlag
                )
            self.AtLeastOne = True
            self.DendArr = DendArr
        else:
            self.DendArr     = []
            self.AtLeastOne      = False


    def on_left_click(self, event) -> None:
        """
        as soon as the user clicked two times on the dendrite
        the calculation of the medial axis path will begin

        Args:
            event: event data

        Returns:

        """
        #flags for zooming pan mode if true clicks are not painted
        zoom_flag = self.SimVars.frame.mpl.toolbox.mode == "zoom rect"
        pan_flag = self.SimVars.frame.mpl.toolbox.mode == "pan/zoom"
        if zoom_flag or pan_flag or not self.AnotherDendFlag or not event.inaxes:
            pass
        else:
            x, y = int(event.xdata), int(event.ydata)
            coords = np.array([y, x])
            self.coords.append(coords)

            self.sc = self.SimVars.frame.mpl.axes.scatter(x, y, marker="x", color="red")
            self.SimVars.frame.mpl.canvas.draw()
            if len(self.coords) == 2:

                self.SimVars.frame.set_status_message.setText(self.SimVars.frame.status_msg["8"])
                self.DendArr.append(Dendrite(self.SimVars.frame.tiff_Arr, SimVars=self.SimVars))
                for Dend in self.DendArr:
                    Dend.thresh = int(self.SimVars.frame.thresh_slider.value())
                self.AtLeastOne=True
                if self.thresh is not False:
                    self.DendArr[-1].thresh = self.thresh
                self.DendArr[-1].coords = self.coords
                self.coords = []
                try:
                    self.DendArr[-1].calc_medial_axis_path()
                except:
                    self.SimVars.frame.set_status_message.setText("A path couldn't be found! Try a different threshold")
                    for artist in self.SimVars.frame.mpl.axes.get_children():
                        if isinstance(artist, mpl.collections.PathCollection):
                            markers = artist.get_offsets()

                            # Remove the last two markers
                            updated_markers = markers[:-2]

                            # Update the marker data
                            artist.set_offsets(updated_markers)
                            self.SimVars.frame.mpl.canvas.draw()
                    self.DendArr.pop()
                self.medial_axis_path_changer(self.DendArr[-1])

                self.DendArr[-1].control_points = (self.DendArr[-1].get_control_points()).astype(int)

                MakeButtonActive(self.SimVars.frame.dendritic_width_button)
                MakeButtonActive(self.SimVars.frame.spine_button)
                MakeButtonActive(self.SimVars.frame.spine_button_NN)
                MakeButtonActive(self.SimVars.frame.measure_puncta_button)
                if(self.SimVars.multitime_flag):
                    self.SimVars.frame.show_stuff([self.SimVars.frame.Dend_shift_check])
    
                self.AnotherDendFlag = True
                self.sc = []

    def medial_axis_path_changer(self,Dend) -> None:
        """
        creates the line interactor for drag and drop the points
        Returns:

        """
        Dend.pol = Polygon(
            Dend.curvature_sampled, fill=False, closed=False, animated=True
        )
        self.SimVars.frame.mpl.axes.add_patch(Dend.pol)
        Dend.lineinteract = LineInteractor(
            self.SimVars.frame.mpl.axes, self.SimVars.frame.mpl.canvas, Dend.pol,not self.AnotherDendFlag
        )

    def on_key_press(self, event):
        """Handles the key press event.

        Args:
            event: The key press event.

        Returns:
            None
        """
        if not event.inaxes:
            return
        elif(event.key == 't' and self.AtLeastOne):
            self.AnotherDendFlag = not self.AnotherDendFlag
            if(self.AnotherDendFlag):
                self.SimVars.frame.add_commands(["MP_Desc","MP_line"])
            else:
                self.SimVars.frame.add_commands(["MP_Desc","MP_vert"])
        elif(event.key == 'd'):
            if(self.AnotherDendFlag):
                self.sc.remove()
                self.coords = []
                self.SimVars.frame.canvas.draw()
        elif event.key == 'backspace':
            self.DendClear(self.SimVars.frame.tiff_Arr)
            MakeButtonInActive(self.SimVars.frame.dendritic_width_button)

    def DendClear(self,tiff_Arr):

        MakeButtonInActive(self.SimVars.frame.spine_button)
        MakeButtonInActive(self.SimVars.frame.spine_button_NN)
        MakeButtonInActive(self.SimVars.frame.measure_puncta_button)
        self.AtLeastOne=False
        self.AnotherDendFlag=True
        self.coords = ([])
        self.SimVars.frame.mpl.clear_plot()
        image = self.SimVars.frame.tiff_Arr[self.SimVars.frame.actual_timestep, self.SimVars.frame.actual_channel, :, :]
        self.SimVars.frame.mpl.update_plot((image>=self.thresh)*image)
        self.click_conn = self.SimVars.frame.mpl.canvas.mpl_connect("button_press_event", self.on_left_click)
        self.press_conn = self.SimVars.frame.mpl.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.DendArr = []
        self.SimVars.frame.add_commands(["MP_Desc","MP_line"])


def DendSave_csv(Dir,Dend_Arr,shift):
    """
    Saves dendrite data to multiple CSV files, each corresponding to a specific channel.

    Args:
        Dir (str): Directory path where the CSV files will be saved.
        Dend_Arr (list): List of dendrite arrays.

    Returns:
        None
    """

    nChans = Dend_Arr[0].dend_lumin.shape[-1]
    nSnaps = Dend_Arr[0].dend_lumin.shape[-2]
    customhead = ['Dendrite']

    oglist = [['Dendrite: '+str(i)]+['Width'] +['Timestep '+ str(i) + ' (Luminosity (mid.))' for i in range(1,nSnaps+1)]
     +['Timestep '+ str(i) + ' (Luminosity (ell.))' for i in range(1,nSnaps+1)] for i in range(len(Dend_Arr))]
    flattened_list = [item for sublist in oglist for item in sublist]
    for c in range(nChans):
        csv_file_path = Dir+'Dendrite_Channel_'+str(c)+'.csv'
        DendVar = []
        for D in Dend_Arr:
            pts   = D.dend_stat[:,:2] + shift
            loc   = np.array([str([x,y]) for x,y in pts])
            width = np.array([str(t*D.WidthFactor) for t in pts])
            x =  np.hstack([loc.reshape(-1,1),width.reshape(-1,1),D.dend_lumin[:,:,c],D.dend_lumin_ell[:,:,c]])
            DendVar.append(x)
        max_sublists = max(len(var) for var in DendVar)
        max_length = max(max(len(sublist) for sublist in var) for var in DendVar)

        for i,var in enumerate(DendVar):
            if(len(var)<max_sublists):
                num_empty_entries = max_sublists - len(var)
                empty_rows = np.empty((num_empty_entries,max_length),dtype=var.dtype)
                DendVar[i] = np.vstack((var,empty_rows))
        flattened_data = []
        # Iterate over the sublists
        for i in range(max_sublists):
            sublist_data = []
            for var in DendVar:
                sublist = var[i] if i < len(var) else []
                sublist_data.extend(sublist)
            flattened_data.append(sublist_data)
        # Write data to CSV file
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(flattened_list)
            for row in flattened_data:
                writer.writerow(row)

def DendSave_json(Dir,Dend_Arr,tf,Snapshots,Channels,Unit,shift):
    """
    Saves dendrite data to multiple CSV files, each corresponding to a specific channel.

    Args:
        Dir (str): Directory path where the CSV files will be saved.
        Dend_Arr (list): List of dendrite arrays.

    Returns:
        None
    """

    DDic = []
    for D in Dend_Arr:
        pts = D.control_points + shift
        Start,End = pts[0].tolist(),pts[-1].tolist()
        length = D.length
        Mean = []
        Area = []
        Max  = []
        Min  = []
        RID  = []
        ID   = []
        for C in range(Channels):
            me,ar,ma,mi,ri,ide = MeasureDend(D.dendritic_surface_matrix,tf[:,C,:,],Unit,Snapshots)

            Mean.append(me)
            Area.append(ar)
            Max.append(ma)
            Min.append(mi)
            RID.append(ri)
            ID.append(ide)
        DDic.append({'Start':Start,'End':End,'Length':length,'Mean':Mean,
                'Area':Area,'Max':Max,'Min':Min,'Raw Integrated Density':RID,
                'Integrated Density':ID})
    with open(Dir + "Dendrites.json", "w") as fp:
        json.dump([D for D in DDic], fp, indent=4)

def DendSave_imj(Dir,Dend_Arr,shift):
    os.mkdir(path=Dir+'ImageJ/')
    Dir2 = Dir+'ImageJ/'
    for i,D in enumerate(Dend_Arr):
        pts = np.flip(D.dend_stat[:,:2]+shift,axis=1)

        roi = rf.ImagejRoi.frompoints(pts)
        roi.roitype = rf.ROI_TYPE.FREELINE
        roi.tofile(Dir2+'DendMed_'+str(i)+'.roi')

        pts = D.contours[0].squeeze()

        roi = rf.ImagejRoi.frompoints(pts)

        roi.tofile(Dir2+'DendSeg_'+str(i)+'.roi')


def MeasureDend(mask, tiff_Arr, Unit,Snapshots):

    """
    Input:
            S (Synapse)
            tiff_Arr (np.array) : The pixel values of the of tiff files
            SimVars  (class)    : The class holding all simulation parameters
    Output:
            None

    Function:
            Finds the relevant places in the tiff file and measures these for each Dendrite
    """
    Mean = []
    area = []
    Max  = []
    Min  = []
    RawIntDen = []
    IntDen = []
    for i in range(Snapshots):
        try:
            roi  = tiff_Arr[i].astype(np.float64)
            roi[np.where(mask == 0)] = math.nan
            area_pix = np.sum(mask)
            area.append(int(area_pix) * Unit**2)
            Max.append(int(np.nanmax(roi)))
            Min.append(int(np.nanmin(roi)))
            RawIntDen.append(int(np.nansum(roi)))
            IntDen.append(np.nansum(roi) * Unit**2)
            Mean.append(np.nanmean(roi))

        except Exception as ex:
            print(ex)
            area.append(math.nan)
            Mean.append(math.nan)
            Max.append(math.nan)
            Min.append(math.nan)
            RawIntDen.append(math.nan)
            IntDen.append(math.nan)

    return Mean,area,Max,Min,RawIntDen,IntDen

