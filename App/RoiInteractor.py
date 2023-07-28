import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.patches import Polygon
from matplotlib.transforms import Bbox

from .Utility import *


class RoiInteractor:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, canvas, poly,loc=None,shift=[],Snapshot=0,nSnaps=1):
        if poly.figure is None:
            raise RuntimeError(
                "You must first add the polygon to a figure "
                "or canvas before defining the interactor"
            )
        self.ax = ax
        self.ax.patch.set_alpha(0.5)
        self.poly = poly
        self.loc  = loc
        self.OgLoc = loc
        x, y = zip(*self.poly.xy)
        self.Snapshot = Snapshot
        self.line = Line2D(
            x,
            y,
            marker="o",
            markerfacecolor="r",
            markersize=1.2 * self.epsilon,
            fillstyle="full",
            linestyle=None,
            linewidth=1.5,
            animated=True,
            antialiased=True,
            alpha=0.75,
        )
        self.NoShift = True
        if(loc is not None):
            if(Snapshot>0):
                self.line_centre = Line2D(
                    [loc[0]],[loc[1]],
                    marker="o",
                    markerfacecolor="k",
                    markersize=1.2 * self.epsilon,
                    fillstyle="full",
                    linestyle=None,
                    linewidth=1.5,
                    color='r',
                    animated=True,
                    antialiased=True,
                    )
            else:
                self.line_centre = Line2D(
                    [loc[0]],[loc[1]],
                    marker="o",
                    markerfacecolor="gray",
                    markersize=1.2 * self.epsilon,
                    fillstyle="full",
                    linestyle=None,
                    linewidth=1.5,
                    color='gray',
                    animated=True,
                    antialiased=True,
                    )
            self.ax.add_line(self.line_centre)
            if(shift is None or len(shift)==0):
                self.shift = [[0,0]]*nSnaps
            else:
                self.shift = shift
            self.points = np.array(self.poly.xy)-np.array(loc)
            self.NoShift = False
        else:
            self.shift = [[0,0]]*nSnaps
        self.ax.add_line(self.line)

        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect("draw_event", self.on_draw)
        canvas.mpl_connect("button_press_event", self.on_button_press)
        canvas.mpl_connect("key_press_event", self.on_key_press)
        canvas.mpl_connect("button_release_event", self.on_button_release)
        canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.canvas = canvas
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def clear(self):
        self.line.set_visible(False)
        self.poly.set_visible(False)
        for p,l in zip(self.ax.patches,self.ax.lines):
            p.remove()
            l.remove()


    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.set_alpha(0.1)
        self.ax.draw_artist(self.line)
        if hasattr(self,"line_centre"): self.ax.draw_artist(self.line_centre)

    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is called."""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state


    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        # I know this can lead to multiple ROIs being selected - possible fix: pass all centres and just select
        # the closest centre
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        (indseq,) = np.nonzero(d == d.min())
        ind = indseq[0]
        if(self.NoShift):
            if d[ind] >= self.epsilon:
                ind = None
            return ind
        else:
            d1 = np.hypot(self.loc[0] - event.xdata, self.loc[1] - event.ydata)
            if d[ind] >= self.epsilon and d1 > self.epsilon:
                ind = None
            elif d1 > d[ind]:
                return ind
            elif d1 < d[ind]:
                return 'mid'

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.showverts:
            return
        if event.button != 1:
            return

        if(isinstance(self._ind,str)):
            self.shift[self.Snapshot] = [round(self.loc[0]-self.OgLoc[0],0),round(self.loc[1]-self.OgLoc[1],0)]
        self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == "t":
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.NoShift: self.line_centre.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == "d":
            ind = self.get_ind_under_point(event)
            if ind is not None and not isinstance(ind,str):
                self.poly.xy = np.delete(self.poly.xy, ind, axis=0)
                self.line.set_data(zip(*self.poly.xy))

        elif event.key == "i":
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i + 1, [event.xdata, event.ydata], axis=0
                    )
                    self.line.set_data(zip(*self.poly.xy))
                    break
        if self.line.stale:
            self.canvas.draw_idle()

    def getPolyXYs(self):
        return self.poly.xy

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if isinstance(self._ind,str): 
            if self.Snapshot>0:
                x, y = event.xdata, event.ydata

                self.canvas.restore_region(self.background)
                self.line_centre.set_data([x], [y])
                self.loc = [x,y]

                self.line.set_data(zip(*self.points+np.array([x,y])))

                self.poly.xy = self.points+np.array([x,y])
                self.ax.draw_artist(self.line_centre)
                self.ax.draw_artist(self.line)
                self.canvas.draw_idle()
                self.canvas.blit(self.ax.bbox)

            return 

        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y

        self.canvas.restore_region(self.background)
        self.line.set_data(zip(*self.poly.xy))
        if self.loc is not None: self.points = np.array(self.poly.xy)-np.array(self.loc)
        for ix in self.ax.patches:
            self.ax.draw_artist(ix)
        self.ax.draw_artist(self.line)
        if not self.NoShift: self.ax.draw_artist(self.line_centre)
        self.canvas.draw_idle()

        self.canvas.blit(self.ax.bbox)

class RoiInteractor_BG:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    epsilon = 10  # max pixel distance to count as a vertex hit

    def __init__(self, ax,canvas,Spine):

        self.bgloc = np.array(Spine.bgloc)
        self.points = np.array(Spine.points)-np.array(Spine.location)
        self.poly = Polygon(self.points+self.bgloc, fill=True, animated=True, alpha=0.5, color="gray")
        self.ax = ax
        self.canvas = canvas
        self.ax.add_patch(self.poly)
        self.ax.patch.set_alpha(0.5)  

        self.line = Line2D(
            [self.bgloc[0]],[self.bgloc[1]],
            marker="o",
            markerfacecolor="r",
            markersize=5,
            fillstyle="full",
            linestyle=None,
            linewidth=1.5,
            animated=True,
            antialiased=True,
            )

        self.ax.add_line(self.line)

        self.cid = self.poly.add_callback(self.poly_changed)

        canvas.mpl_connect("draw_event", self.on_draw)
        canvas.mpl_connect("button_press_event", self.on_button_press)
        canvas.mpl_connect("button_release_event", self.on_button_release)
        canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.canvas.draw()



    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.ax.set_alpha(0.1)

    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is called."""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state


    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        d = np.linalg.norm([self.line.get_data()[0][0]-event.xdata,self.line.get_data()[1][0]-event.ydata])
        if d >= self.epsilon:
            ind = None
        else:
            ind = 1

        return ind

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def clear(self):
        self.line.set_visible(False)
        self.poly.set_visible(False)
        for p,l in zip(self.ax.patches,self.ax.lines):
            p.remove()
            l.remove()
            
    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.button != 1:
            return
        self._ind = None

        self.canvas.draw()

    def getPolyXYs(self):
        return self.poly.xy

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if(self._ind is not None):
            x, y = event.xdata, event.ydata


            self.line.set_data([x], [y])
            self.poly.xy = np.array([x,y]) + np.array(self.points)

            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.poly)
            self.ax.set_alpha(0.1)
            self.ax.draw_artist(self.line)
            self.canvas.draw_idle()
            self.canvas.blit(self.ax.bbox)

