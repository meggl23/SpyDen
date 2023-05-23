from LineInteractor import *
from MPL_Widget import *

from scipy.signal import medfilt2d

from PathFinding import (
    medial_axis_path,
    downsampling_max_pool,
    curvature_polygon,
    curvature_eval,
)

from matplotlib.patches import Polygon

from scipy.ndimage import gaussian_filter1d, gaussian_filter
import cv2 as cv
from skimage.draw import ellipse
from PathFinding import GetAllpointsonPath

from to_pyqt import MakeButtonActive


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

    def set_surface_contours(self, max_neighbours: int = 6, sigma: int = 10) -> None:
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
        smoothed_all_pts = np.stack((gaussian_y, gaussian_x), axis=1)
        actual_image = self.tiff_arr[self.actual_timestep, self.actual_channel, :, :]
        median = medfilt2d(actual_image, kernel_size=5)
        print(self.thresh)
        if self.thresh is not False:
            median_thresh = median >= self.thresh

        else:
            median_thresh = median >= np.mean(median)
        width_arr, degrees = getWidthnew(
            median_thresh,
            smoothed_all_pts,
            sigma=sigma,
            max_neighbours=max_neighbours,
        )
        mask = np.zeros(shape=self.Morphologie.shape)

        self.dend_stat = np.zeros(shape= (len(smoothed_all_pts), 5))
        if(self.SimVars.multitime_flag):
            print('here')
            Snaps = self.SimVars.Snapshots
        else:
            Snaps = 1
        if(self.SimVars.multiwindow_flag):
            Chans = self.SimVars.Channels
        else:
            Chans = 1
        self.dend_lumin = np.zeros(shape= (len(smoothed_all_pts), Snaps,Chans))
        for pdx, p in enumerate(smoothed_all_pts):
            self.dend_stat[pdx, 0] = p[1]
            self.dend_stat[pdx, 1] = p[0]
            self.dend_stat[pdx, 2] = width_arr[pdx]
            self.dend_stat[pdx, 3] = 2
            self.dend_stat[pdx, 4] = degrees[pdx]


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

        gaussian_mask = (gaussian_filter(input=mask, sigma=4) >= np.mean(mask)).astype(np.uint8)
        self.contours, _ = cv.findContours(gaussian_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        self.dendritic_surface_matrix= gaussian_mask

    def get_contours(self) -> tuple:
        return self.contours

    def get_dendritic_surface_matrix(self) -> np.ndarray:
        return self.dendritic_surface_matrix


class DendriteMeasurement:
    """this class handles the communication between the gui and the data calculation classes
    this class is furthermore used to add functionality to the gui classes"""

    def __init__(self, SimVars, tiff_Arr,DendArr=None):

        self.coords = []
        self.SimVars = SimVars
        self.canvas = SimVars.frame.mpl.canvas
        self.axis = self.SimVars.frame.mpl.axes
        self.tiff_Arr   = tiff_Arr
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
        if zoom_flag or pan_flag or not self.AnotherDendFlag:
            pass
        else:
            x, y = int(event.xdata), int(event.ydata)
            coords = np.array([y, x])
            self.coords.append(coords)

            self.sc = self.axis.scatter(x, y, marker="x", color="red")
            self.canvas.draw()

            if len(self.coords) == 2:

                self.SimVars.frame.set_status_message.setText(self.SimVars.frame.status_msg["8"])
                self.DendArr.append(Dendrite(self.tiff_Arr, SimVars=self.SimVars))
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
                    for artist in self.axis.get_children():
                        if isinstance(artist, mpl.collections.PathCollection):
                            markers = artist.get_offsets()

                            # Remove the last two markers
                            updated_markers = markers[:-2]

                            # Update the marker data
                            artist.set_offsets(updated_markers)
                            self.canvas.draw()
                    self.DendArr.pop()
                self.medial_axis_path_changer(self.DendArr[-1])

                self.DendArr[-1].control_points = (self.DendArr[-1].get_control_points()).astype(int)

                MakeButtonActive(self.SimVars.frame.dendritic_width_button)
                MakeButtonActive(self.SimVars.frame.spine_button)
                MakeButtonActive(self.SimVars.frame.spine_button_NN)
                MakeButtonActive(self.SimVars.frame.delete_old_result_button)

                MakeButtonActive(self.SimVars.frame.measure_puncta_button)
    
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
        elif event.key == 'backspace':
            self.AtLeastOne=False
            self.AnotherDendFlag=True
            self.coords = ([])
            self.SimVars.frame.mpl.clear_plot()
            self.SimVars.frame.dend_thresh()
            self.canvas = self.SimVars.frame.mpl.canvas
            self.axis = self.SimVars.frame.mpl.axes
            self.click_conn = self.SimVars.frame.mpl.canvas.mpl_connect("button_press_event", self.on_left_click)
            self.press_conn = self.SimVars.frame.mpl.canvas.mpl_connect("key_press_event", self.on_key_press)
            self.DendArr = []
            self.SimVars.frame.add_commands(["MP_Desc","MP_line"])
