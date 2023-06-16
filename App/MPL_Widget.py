import PyQt5.QtCore

from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as Navi,
    FigureCanvasQT,
)

from typing import Callable

import numpy as np

import traceback


class MyPaintWidget(QWidget):
    """
    class that creates a QWidget frot Matplotlib Canvas
     so that the Widget can be added easily to a QT Layout

    """

    def __init__(self, image: np.ndarray, instance: object):
        super().__init__()
        self.image = image
        self.instance = instance
        self.Start_plot()


    def Start_plot(self) -> None:
        self.figure = plt.figure(figsize=(14, 14),dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.instance.grid.addWidget(
            self.canvas, 6, 2, 13, 10
        ) 
        self.toolbox = Navi(self.canvas, self.instance)
        self.instance.grid.addWidget(self.toolbox, 19, 2, 1, 10)
        self.axes.imshow(self.image)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        self.figure.tight_layout()

    def remove_plot(self) -> None:
        """
        remove plot from widget
        Returns:

        """
        self.instance.grid.removeWidget(self.canvas)
        self.instance.grid.removeWidget(self.toolbox)
        self.toolbox.deleteLater()
        self.axes.clear()
        self.figure.clear()
        self.canvas.draw()
        plt.close()

    def clear_plot(self) -> None:
        """
        remove plot from widget
        Returns:
        """
        
        self.remove_plot()
        self.Start_plot()

        return self.canvas

    def update_plot(self, image) -> None:
        """
        updates the figure in the matplotlib Figure
        Returns:None

        """
        self.axes.imshow(image)
        self.canvas.draw()

    def connect_to_clicker_event(self, clicker_type: str, func: Callable) -> None:
        cid = self.canvas.mpl_connect(clicker_type, func)
        return cid

def debug_trace():
  '''Set a tracepoint in the Python debugger that works with Qt'''

  from pdb import set_trace
  pyqtRemoveInputHook()
  set_trace()
