import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100

from .Utility import MakeButtonInActive,MakeButtonActive

class Synapse(object):
    """Class that holds the parameters associated with the chosen synapse"""
    def __init__(self,loc,bgloc,pts=[],dist=None,Syntype=None,shift=[],channel=0,local_bg=0,closest_Dend=0):
        self.type = Syntype
        self.location = loc
        self.bgloc = bgloc

        self.mean = []
        self.min = []
        self.max = []
        self.RawIntDen = []
        self.IntDen = []
        self.local_bg = []

        self.area = []
        if dist == None:
            self.distance = 0
        else:
            self.distance = dist

        self.points = pts

        self.shift = shift
        self.channel = channel
        self.closest_Dend = closest_Dend

class Spine_Marker:

    Epsilon = 10

    def __init__(self, SimVars, points=np.array([]), scores=np.array([]),flags = np.array([])):

        self.points  = points
        self.scores  = scores
        self.flags   = flags.astype(int)
        self.SimVars = SimVars
        self.colors = ['red', 'green','yellow']

        self.shift_is_held = False
        self.control_is_held = False
        self.butt_press = self.SimVars.frame.mpl.canvas.mpl_connect(
            "button_press_event", self._on_left_click
        )
        self.key_press = self.SimVars.frame.mpl.canvas.mpl_connect(
            "key_press_event", self._on_key_press
        )
        self.on_key_release = self.SimVars.frame.mpl.canvas.mpl_connect(
            "key_release_event", self._on_key_release
        )
        self.canvas = self.SimVars.frame.mpl.canvas
        self.axes = self.SimVars.frame.mpl.axes
        self.scatter = None
        self.draw_points()

    def draw_points(self):
        """Draws the points on the plot.

        If no points are available, the canvas is drawn without any points.
        If points are available and no scatter plot exists, a new scatter plot is created and drawn.
        If points are available and a scatter plot already exists, the existing scatter plot is updated and redrawn.

        Returns:
            None
        """
        if len(self.points) == 0:
            self.canvas.draw()
            MakeButtonInActive(self.SimVars.frame.spine_button_ROI)
        elif len(self.points) > 0 and self.scatter is None:
            MakeButtonActive(self.SimVars.frame.spine_button_ROI)
            self.scatter = self.axes.scatter(
                self.points[:, 0], self.points[:, 1], marker="x", c=[self.colors[self.flag] for self.flag in self.flags]
            )
            self.canvas.draw()
        else:
            MakeButtonActive(self.SimVars.frame.spine_button_ROI)
            self.scatter.set_visible(False)
            self.canvas.draw()
            self.scatter = self.axes.scatter(
                self.points[:, 0], self.points[:, 1], marker="x", c=[self.colors[self.flag] for self.flag in self.flags]
            )
            self.canvas.draw()

    def disconnect(self):
        """Disconnects the event listeners from the canvas.

        Removes the connections between the button press event and the key press event
        and their corresponding event handlers.

        Returns:
            None
        """
        self.SimVars.frame.mpl.canvas.mpl_disconnect(self.butt_press)
        self.SimVars.frame.mpl.canvas.mpl_disconnect(self.key_press)

    def _on_key_press(self, event):
        """Event handler for key press events.

        Performs different actions based on the key pressed. If the zoom or pan/zoom
        mode is active, no action is taken. If the backspace key is pressed, the stored
        points, scores, and flags are cleared. If the shift key is pressed, the shift_is_held
        flag is set to True. If the control key is pressed, the control_is_held flag is set
        to True. If a data point is within a certain distance of the pressed key point when 
        d is pressed, it is removed from the stored points, scores, and flags. If there are 
        no more points, the scatter plot is made invisible.

        Args:
            event: The key press event.

        Returns:
            None
        """
        zoom_flag = self.SimVars.frame.mpl.toolbox.mode == "zoom rect"
        pan_flag = self.SimVars.frame.mpl.toolbox.mode == "pan/zoom"
        if(hasattr(self.SimVars,'points_NN')):
            nn_points = self.SimVars.points_NN
            nn_scores = self.SimVars.scores_NN
            nn_flags  = self.SimVars.flags_NN
        if zoom_flag or pan_flag:
            pass
        else:
            if(event.key == "backspace"):
                self.points = np.array([])
                self.scores = np.array([])
                self.flags = np.array([])
                if(hasattr(self.SimVars,'points_NN')):
                    del self.SimVars.points_NN
                    del self.SimVars.flags_NN
                    del self.SimVars.scores_NN
            if(event.key == "shift"):
                self.shift_is_held = True
            if(event.key == "control"):
                self.control_is_held = True
            key_press_point = np.array([event.xdata,event.ydata]).reshape(1,2)
            if len(self.points) > 0:

                dist_values = np.sqrt(np.sum((self.points - key_press_point) ** 2, axis=1))
                index,val = np.argmin(dist_values),np.min(dist_values)
                if val <= Spine_Marker.Epsilon and event.key == "d":
                    self.points = np.delete(
                        self.points, [index * 2, index * 2 + 1]
                    ).reshape(-1, 2)
                    self.scores = np.delete(self.scores,index)
                    self.flags = np.delete(self.flags,index)
                    if(hasattr(self.SimVars,'points_NN')):
                        self.SimVars.points_NN = np.delete(nn_points[nn_scores>self.SimVars.frame.ml_confidence_slider.value()/10], 
                                                [index * 2, index * 2 + 1]).reshape(-1, 2)
                        self.SimVars.scores_NN = np.delete(nn_scores[nn_scores>self.SimVars.frame.ml_confidence_slider.value()/10],index)
                        self.SimVars.flags_NN = np.delete(nn_flags[nn_scores>self.SimVars.frame.ml_confidence_slider.value()/10],index)
            if len(self.points) == 0:
                self.scatter.set_visible(False)
            self.draw_points()

    def _on_key_release(self, event):
        """Event handler for key release events.

        Resets the shift_is_held flag to False when the shift key is released.
        Resets the control_is_held flag to False when the control key is released.

        Args:
            event: The key release event.

        Returns:
            None
        """
        if event.key == "shift":        
            self.shift_is_held = False
        if event.key == "control":
            self.control_is_held = False

    def _on_left_click(self, event):
        """Event handler for left click events.

        Adds points to the scatter plot based on the coordinates of the left click event.
        Modifies the points, scores, and flags arrays accordingly if shift/control is held.
        Updates the scatter plot with the new points.

        Args:
            event: The left click event.

        Returns:
            None
        """
        zoom_flag = self.SimVars.frame.mpl.toolbox.mode == "zoom rect"
        pan_flag = self.SimVars.frame.mpl.toolbox.mode == "pan/zoom"

        if zoom_flag or pan_flag or not event.inaxes:
            pass
        else:
            coords = np.array([event.xdata, event.ydata]).reshape(1, 2)
            if len(self.points) == 0:
                self.points = coords
                self.scores = np.array([1])
                if(self.shift_is_held):
                    self.flags = np.array([1])
                else:
                    self.flags = np.array([0])
            else:
                New = True
                if(self.shift_is_held):
                    if(np.linalg.norm(self.points-coords,axis=-1)<10).any():
                        indx1 = np.argmin(np.linalg.norm(self.points-coords,axis=-1))
                        self.flags[indx1] = 1
                        if(hasattr(self.SimVars,'points_NN')):
                            indx2 = np.argmin(np.linalg.norm(self.SimVars.points_NN-coords,axis=-1))
                            self.SimVars.flags_NN[indx2] = 1

                        New = False
                    else:
                        self.points = np.append(self.points, coords, axis=0)
                        self.scores = np.append(self.scores,1)
                        self.flags  = np.append(self.flags,1)
                elif(self.control_is_held):
                    if(np.linalg.norm(self.points-coords,axis=-1)<10).any():
                        indx1 = np.argmin(np.linalg.norm(self.points-coords,axis=-1))
                        self.flags[indx1] = 2
                        if(hasattr(self.SimVars,'points_NN')):
                            indx2 = np.argmin(np.linalg.norm(self.SimVars.points_NN-coords,axis=-1))
                            self.SimVars.flags_NN[indx2] = 2

                        New = False
                    else:
                        self.points = np.append(self.points, coords, axis=0)
                        self.scores = np.append(self.scores,1)
                        self.flags  = np.append(self.flags,2)
                else:
                    self.points = np.append(self.points, coords, axis=0)
                    self.scores = np.append(self.scores,1)
                    self.flags  = np.append(self.flags,0)
            if(hasattr(self.SimVars,'points_NN')):
                if(New):
                    self.SimVars.points_NN = np.append(self.SimVars.points_NN, coords, axis=0)
                    self.SimVars.scores_NN = np.append(self.SimVars.scores_NN,self.scores[-1])
                    self.SimVars.flags_NN  = np.append(self.SimVars.flags_NN,self.flags[-1])


            self.draw_points()
