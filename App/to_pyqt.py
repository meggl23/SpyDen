import os
import shutil
import glob

import numpy as np

from .MPL_Widget import *
from .DataRead import *
from matplotlib.widgets import Slider, Button
from matplotlib.path import Path

from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QPainter,QDesktopServices
from PyQt5.QtCore import QUrl

from . import GenFolderStruct as GFS

from .SynapseFuncs import ROI_And_Neck
from .RoiInteractor import RoiInteractor,RoiInteractor_BG
from .PunctaDetection import save_puncta,PunctaDetection,Puncta
from .PathFinding import GetLength

from superqt import QLabeledRangeSlider
import webbrowser as wb
import platform
import time

DevMode = False
version = '1.1.0'

def catch_exceptions(func):

    """Decorator to catch and handle exceptions raised by a function.

    The decorator wraps the provided function with exception handling logic.
    It catches any exceptions raised by the function and sets an error message
    in the status message field.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    def func_wrapper(*args, **kwargs):
        global DevMode
        try:
            self = args[0]
            return func(self)
        except TypeError as e:
            try:
                return func(*args,*kwargs)
            except Exception as e:
                self.set_status_message.setText("This went wrong: " + str(e))
                if DevMode:
                    raise
        except Exception as e:
            self.set_status_message.setText("This went wrong: " + str(e))
            if DevMode:
                raise
    return func_wrapper


def handle_exceptions(cls):
    """Decorates the methods of a class to catch and handle exceptions.

    The function iterates over the methods of the provided class and decorates
    each method with the `catch_exceptions` decorator, which catches and handles
    any exceptions raised by the method.

    Args:
        cls (class): The class whose methods will be decorated.

    Returns:
        class: The class with decorated methods.
    """

    for name, method in vars(cls).items():
        if callable(method) and not name.startswith("__"):
            setattr(cls, name, catch_exceptions(method))
    return cls


class ClickSlider(QSlider):
    """
    Custom slider class that allows setting the value by clicking on the slider.

    Inherits from QSlider class.

    Methods:
        mousePressEvent(e):
            Handles the mouse press event for the slider. If the left mouse button is pressed,
            calculates the corresponding value based on the click position and sets it as the current value.
            Accepts the event to indicate that it has been handled.

        mouseReleaseEvent(e):
            Handles the mouse release event for the slider. If the left mouse button is released,
            calculates the corresponding value based on the release position and sets it as the current value.
            Accepts the event to indicate that it has been handled.

    Usage:
        slider = ClickSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(50)
        slider.valueChanged.connect(my_function)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton or e.button() == Qt.RightButton:
            e.accept()
            x = e.pos().x()
            value = round((self.maximum() - self.minimum()) * x / self.width() + self.minimum())
            self.setValue(value)
        else:
            return super().mousePressEvent(self, e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton or e.button() == Qt.RightButton:
            e.accept()
            x = e.pos().x()
            value = round((self.maximum() - self.minimum()) * x / self.width() + self.minimum())
            self.setValue(value)
        else:
            return super().mouseReleaseEvent(self, e)

@handle_exceptions
class DataReadWindow(QWidget):
    """
    class that makes the Data Read Window
    """

    def __init__(self):
        super().__init__()

        self.title = "Data Read Window"
        self.left = 100
        self.top = 100
        self.width = 1500
        self.height = 1400
        self.actual_channel = 0
        self.tiff_Arr = np.zeros(shape=(4, 4, 1024, 1024))
        self.number_channels = self.tiff_Arr.shape[1]
        self.number_timesteps = self.tiff_Arr.shape[0]
        self.actual_timestep = 0
        self.punctas = []
        self.PunctaCalc = False
        self.SpinesMeasured = False

        home_dir = os.path.expanduser("~")

        # Specify the folders
        folder_name = 'SpydenML'
        subfolder_name = 'SynapseMLModel'


        # Create the full path
        folder_path = os.path.expanduser("~")
        self.NN_path = os.path.join(home_dir, folder_name, subfolder_name)
        self.default_ML_address = os.path.join(home_dir, folder_name)


        if(os.path.exists(self.NN_path)):
            self.NN = True
        else:
            self.NN = False

        self.SimVars = Simulation(0, 0, 0, 0,0,0,frame=self)
        self.status_msg = {
            "0": "Select Folder Path",
            "1": "Set Parameters",
            "2": "Medial Axis Path Calculation",
            "3": "Mark the synapses by a click",
            "4": "Something went wrong",
            "5": "",
            "6": "Dendrite and spine roi's have to be calculated to save results",
            "7": "Everything was saved properly",
            "8": "Medial Axis Path Calculation",
            "9": "Change the ROIs of the spines via drag and drop of the points",
            "10": "Old data was loaded",
            "11": "Calculating puncta"
        }

        self.command_list = {
        
        "MP_Desc":"Generate and edit the medial axis paths of dendrites. After every two clicks a path is generated \n"
        ,
        "MP_line": "Dendrite medial axis calculation: \n"+
        "  - p                  - toggle pan \n" +
        "  - o                  - toggle rectangle to zoom \n" +
        "  - h                  - reset view \n" +
        "  - t                  - toggle between dragable vertices and lines \n" +
        "  - Left click    - mark the start and the end of the dendrite \n" + 
        "  - d                 - delete the first marker \n" +
        "  - backspace - clear the image \n" + 
        "The slider can be used to set the threshold of the image"
        ,
        "MP_vert":"Edit the dendrite path with draggable nodes:\n"+
        "  - p                  - toggle pan \n" +
        "  - o                  - toggle rectangle to zoom \n" +
        "  - h                  - reset view \n" +
        "  - t                  - toggle between dragable vertices and lines \n" +
        "  - d                 - delete a node\n" +
        "  - i                   - insert a Node \n"+
        "  - backspace - clear the image \n"
        "The slider can be used to set the threshold of the image"
        ,
        "Width_Desc":"Change the Dendritic Width Slider to adjust the Dendritic Width(s)",
        "Spine_Desc":"Generate spine locations either by clicking or using a neural network \n",
        "Spine_Func":"Move a Spine location and Press: \n"+
        "  - p                  - toggle pan \n" +
        "  - o                  - toggle rectangle to zoom \n" +
        "  - h                  - reset view \n" +
        "  - Left Click - to mark a Spine \n" +
        "  - d - to delete a marked Spine \n" + 
        "  - backspace - clear the image \n" + 
        "Hold shift (green) or control (yellow, for somatic ROIs) and click to mark special spines (up to 3 different types of spines).",
        "NN_Conf":"Change the Confidence Slider to change the confidence of the NN",
        "SpineROI_Desc":"Edit the ROIs generated by the algorithm via draggable nodes\n",
        "SpineROI_Func":"Commands \n" +         
        "  - p                  - toggle pan \n" +
        "  - o                  - toggle rectangle to zoom \n" +
        "  - h                  - reset view \n" +
        "  - t                  - toggle between dragable vertices and lines \n" +
        "  - d                 - delete a node\n" +
        "  - i                   - insert a Node \n"+
        "  - backspace - clear the image \n"
        "The tolerance slider makes the ROIs bigger or smaller \n"
        "The sigma slider refers to the variance of the smoothing filter: smaller is for small sharp lines, larger for larger blurred lines",
        "SpineBG_Desc":"Edit the locations of the spine ROIs by dragging the red points for the local background calculation\n",
        "Puncta":"Use the sliders to determine the detection threshold for dendritic and synaptic puncta. Puncta are automatically calculated on all channels and snapshots. It is possible that no puncta are found. \n"+
        "  - p                  - toggle pan \n" +
        "  - o                  - toggle rectangle to zoom \n" +
        "  - h                  - reset view \n" 
        }

        self.folderpath = "None"
        self.initUI()

    def initUI(self):

        """
        Initializes the user interface for the application.

        This method sets up various UI elements such as buttons, sliders, labels, and text fields.
        It configures the layout, connects signals to slots, and sets initial values for the UI components.

        - Window title and geometry are set.
        - Window icon is set using an image file.
        - Main layout is set to a grid layout.
        - Button for selecting a folder path is created and connected to the appropriate function.
        - Label displays the selected folder path.
        - Text fields for entering cell and resolution values are created and connected to handle editing events.
        - Combo boxes for selecting projection and transformation options are created and connected to handle selection changes.

        Returns:
            None
        """
   
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        if(platform.system()=='Windows'):
            self.setWindowIcon(QIcon('App\\Brain.ico'))
        else:
            self.setWindowIcon(QIcon(QPixmap("brain.png")))
        self.grid = QGridLayout()
        self.setLayout(self.grid)  # Set the grid layout for DataReadWindow

        self.mpl = MyPaintWidget(self.tiff_Arr[0, 0, :, :], self)


        #============= path button ==================
        self.folderpath_button = QPushButton(self)
        self.folderpath_button.setText("Select Path!")

        MakeButtonActive(self.folderpath_button)
        self.folderpath_button.clicked.connect(self.get_path)
        self.grid.addWidget(self.folderpath_button, 0, 0, 1, 2)
        self.folderpath_label = QLineEdit(self)
        self.folderpath_label.setReadOnly(True)
        self.folderpath_label.setText(str(self.folderpath))
        self.grid.addWidget(self.folderpath_label, 0, 2, 1, 10)
        self.folderpath_button.setToolTip('Provide the path to the folder holding the cell_* folders')

        #============= path input ==================

        self.cell = QComboBox(self)
        self.cell.setEnabled(False)
        self.cell.currentTextChanged.connect(lambda: self.handle_editing_finished(0))
        self.grid.addWidget(self.cell, 1, 1, 1, 1)
        self.grid.addWidget(QLabel("Filename:"), 1, 0, 1, 1)
        self.cell.setToolTip('Select the folder you want to analyze')

        #========= projection dropdown ===============
        self.projection = QComboBox(self)
        choices = ["Max","Min","Mean","Sum","Median","Std"]
        for choice in choices:
            self.projection.addItem(choice)
        self.grid.addWidget(self.projection, 3, 1, 1, 1)
        self.grid.addWidget(QLabel("Projection"), 3, 0, 1, 1)
        self.projection.setEnabled(False)
        self.projection.currentTextChanged.connect(self.on_projection_changed)
        self.projection.setToolTip('z-stack projection')

        #========= analyze dropdown ================
        self.analyze = QComboBox(self)
        choices = ["Luminosity", "Area"]
        for choice in choices:
            self.analyze.addItem(choice)
        self.grid.addWidget(self.analyze, 4, 1, 1, 1)
        self.grid.addWidget(QLabel("Analyze"), 4, 0, 1, 1)
        self.analyze.setEnabled(False)
        self.analyze.currentTextChanged.connect(self.on_analyze_changed)
        self.analyze.setToolTip('Synapse temporal analysis mode')

        #========= multichannel checkbox ================
        self.multiwindow_check = QCheckBox(self)
        self.multiwindow_check.setText("Multi Channel")
        self.multiwindow_check.setEnabled(False)
        self.SimVars.multiwindow_flag  = True
        self.grid.addWidget(self.multiwindow_check, 2, 0, 1, 1)
        self.multiwindow_check.setToolTip('Check if you want to consider all channels simulatenously')

        #========= multitime checkbox ================
        self.multitime_check = QCheckBox(self)
        self.multitime_check.setText("Multi Time")
        self.multitime_check.setEnabled(False)
        self.SimVars.multitime_flag  = False
        self.grid.addWidget(self.multitime_check, 2, 1, 1, 1)
        self.multitime_check.setToolTip('Check if you want to include temporal dynamics')

        #========= resolution input ================
        self.res = QLineEdit(self)
        self.res.setEnabled(False)
        self.grid.addWidget(self.res, 5,1,1,1)
        self.grid.addWidget(QLabel("Resolution (\u03BCm /pixel)"), 5, 0, 1, 1)
        self.res.editingFinished.connect(lambda: self.handle_editing_finished(1))
        
        #========= channel slider ================
        self.channel_label = QLabel("Channel")
        self.grid.addWidget(self.channel_label, 1, 8, 1, 1)
        self.channel_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        #self.channel_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.channel_slider, 1, 2, 1, 6)
        self.channel_slider.setMinimum(0)
        self.channel_slider.setMaximum(self.number_channels - 1)
        self.channel_slider.singleStep()
        self.channel_slider.valueChanged.connect(self.change_channel)
        self.channel_counter = QLabel(str(self.channel_slider.value()))
        self.grid.addWidget(self.channel_counter, 1, 9, 1, 1)
        self.hide_stuff([self.channel_slider,self.channel_counter,self.channel_label])
        self.channel_slider.setToolTip('Scroll through the experimental channels')

        #========= timestep slider ================
        self.timestep_label = QLabel("Timestep")
        self.grid.addWidget(self.timestep_label,2, 8, 1, 1)
        self.timestep_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        #self.timestep_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.timestep_slider, 2, 2, 1, 6)
        self.timestep_slider.setMinimum(0)
        self.timestep_slider.setMaximum(self.number_timesteps - 1)
        self.timestep_slider.setSingleStep(1)
        self.timestep_slider.valueChanged.connect(self.change_channel)
        self.timestep_counter = QLabel(str(self.timestep_slider.value()))
        self.grid.addWidget(self.timestep_counter, 2, 9, 1, 1)
        self.hide_stuff([self.timestep_slider,self.timestep_counter,self.timestep_label])
        self.timestep_slider.setToolTip('Scroll through the experimental timepoints')

        label = QLabel("Dendrite analysis")
        label.setAlignment(Qt.AlignCenter)  # Centers the label horizontally
        label.setStyleSheet("font-size: 18px;")  # Increases the font size

        self.grid.addWidget(label, 6, 0, 1, 2)
        #============= dendritic path button ==================
        self.medial_axis_path_button= QPushButton(self)
        self.medial_axis_path_button.setText(" Calculate Medial Axis")
        MakeButtonInActive(self.medial_axis_path_button)
        self.grid.addWidget(self.medial_axis_path_button, 7, 0, 1, 2)
        self.medial_axis_path_button.clicked.connect(self.medial_axis_eval_handle)
        self.medial_axis_path_button.setToolTip('Calculate the medial axis paths by selecting the start and end of the dendrites')

        #============= dendritic width button ==================
        self.dendritic_width_button = QPushButton(self)
        self.dendritic_width_button.setText("Calculate Dendritic Width")
        MakeButtonInActive(self.dendritic_width_button)
        self.grid.addWidget(self.dendritic_width_button, 8, 0, 1, 2)
        self.dendritic_width_button.clicked.connect(self.dendritic_width_eval)

        label = QLabel("Synapse/Soma analysis")
        label.setAlignment(Qt.AlignCenter)  # Centers the label horizontally
        label.setStyleSheet("font-size: 18px;")  # Increases the font size
        self.grid.addWidget(label, 9, 0, 1, 2)
        #============= manual spine button ==================
        self.spine_button = QPushButton(self)
        self.spine_button.setText("Spine Localization Manually")
        MakeButtonInActive(self.spine_button)
        self.grid.addWidget(self.spine_button, 10, 0, 1, 2)
        self.spine_button.clicked.connect(self.spine_eval_handle)

        #============= NN spine button ==================
        self.button_set_NN = QPushButton(self)
        if(self.NN):
            self.button_set_NN.setText("Set NN (default)")
            self.button_set_NN.clicked.connect(self.set_NN)
        else:
            self.button_set_NN.setText("Download NN")
            self.button_set_NN.clicked.connect(self.download_NN)
        MakeButtonActive(self.button_set_NN)
        self.grid.addWidget(self.button_set_NN, 11, 0, 1, 1)
        self.button_set_NN.setToolTip('Download the online NN or use a local version')


        #============= NN spine button ==================
        self.spine_button_NN = QPushButton(self)
        self.spine_button_NN.setText("Spine Localization via NN")
        MakeButtonInActive(self.spine_button_NN)
        self.grid.addWidget(self.spine_button_NN, 11, 1, 1, 1)
        self.spine_button_NN.clicked.connect(self.spine_NN)


        #============= spine ROI button ==================
        self.spine_button_ROI = QPushButton(self)
        self.spine_button_ROI.setText("Calculate Spine ROI's")
        MakeButtonInActive(self.spine_button_ROI)
        self.grid.addWidget(self.spine_button_ROI, 12, 0, 1, 1)
        self.spine_button_ROI.clicked.connect(self.spine_ROI_eval)

        #============= load ROI button ==================
        self.old_ROI_button = QPushButton(self)
        self.old_ROI_button.setText("Load old ROIs")
        self.grid.addWidget(self.old_ROI_button, 12, 1, 1, 1)
        self.old_ROI_button.clicked.connect(self.old_ROI_eval)
        MakeButtonInActive(self.old_ROI_button)


        #============= spine ROI button ==================
        self.measure_spine_button = QPushButton(self)
        self.measure_spine_button.setText("Measure ROIs")
        MakeButtonInActive(self.measure_spine_button)
        self.grid.addWidget(self.measure_spine_button, 13, 1, 1, 1)
        self.measure_spine_button.clicked.connect(self.spine_measure)

        #============= spine bg button ==================
        self.spine_bg_button = QPushButton(self)
        self.spine_bg_button.setText("Measure local background")
        MakeButtonInActive(self.spine_bg_button)
        self.grid.addWidget(self.spine_bg_button, 13, 0, 1, 1)
        self.spine_bg_button.clicked.connect(self.spine_bg_measure)


        label = QLabel("Puncta analysis")
        label.setAlignment(Qt.AlignCenter)  # Centers the label horizontally
        label.setStyleSheet("font-size: 18px;")  # Increases the font size
        self.grid.addWidget(label, 14, 0, 1, 2)

        #============= puncta button ==================
        self.measure_puncta_button = QPushButton(self)
        self.measure_puncta_button.setText("Get and measure puncta")
        MakeButtonInActive(self.measure_puncta_button)
        self.grid.addWidget(self.measure_puncta_button, 15, 0, 1, 2)
        self.measure_puncta_button.clicked.connect(self.get_puncta)
        self.measure_puncta_button.setToolTip('Calculate puncta for dendritic ROI and all synaptic ROIs')

        label = QLabel("Save/clear")
        label.setAlignment(Qt.AlignCenter)  # Centers the label horizontally
        label.setStyleSheet("font-size: 18px;")  # Increases the font size
        self.grid.addWidget(label, 16, 0, 1, 2)

        #============= delete button ==================
        self.delete_old_result_button = QPushButton(self)
        self.delete_old_result_button.setText("Clear old")
        self.grid.addWidget(self.delete_old_result_button, 17, 0, 1, 1)
        self.delete_old_result_button.clicked.connect(lambda: self.clear_old())
        MakeButtonInActive(self.delete_old_result_button)
        self.delete_old_result_button.setToolTip('Clear old analysis and start fresh')

        self.save_button = QPushButton(self)
        self.save_button.setText("Save results")
        self.grid.addWidget(self.save_button, 17, 1, 1, 1)
        self.save_button.clicked.connect(self.save_results)
        MakeButtonInActive(self.save_button)

        #============= dialog field (status) ==================
        self.set_status_message = QLineEdit(self)
        self.set_status_message.setReadOnly(True)
        self.grid.addWidget(self.set_status_message, 18, 0, 1, 2)
        self.grid.addWidget
        self.set_status_message.setText(self.status_msg["0"])
        self.set_status_message.setToolTip('The current mode you are in (or if something went wrong, the error message)')

        #============= dialog fields (commands) ==================
        self.command_box = QPlainTextEdit(self)
        self.command_box.setReadOnly(True)
        self.grid.addWidget(self.command_box, 19, 0, 1, 2)
        self.command_box.setFixedWidth(550)
        self.command_box.setFixedHeight(100)
        self.command_box.setToolTip('What you can do in your current mode')

        #============= threshold slider ==================
        self.thresh_label = QLabel("Threshold Value")
        self.grid.addWidget(self.thresh_label, 3, 8, 1, 1)
        self.thresh_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.thresh_slider, 3, 2, 1, 6)
        self.hide_stuff([self.thresh_slider,self.thresh_label])

        self.thresh_slider.setToolTip('Select the threshold to filter out background noise')

        #============= dend width slider ==================
        self.neighbour_label = QLabel("Dendritic Width Smoothness")
        self.grid.addWidget(self.neighbour_label, 3, 8, 1, 1)
        self.neighbour_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.neighbour_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.neighbour_slider, 3, 2, 1, 6)
        self.neighbour_slider.setMinimum(1)
        self.neighbour_slider.setMaximum(10)
        self.neighbour_slider.setValue(6)
        self.neighbour_slider.singleStep()
        self.neighbour_counter = QLabel(str(self.neighbour_slider.value()))
        self.grid.addWidget(self.neighbour_counter, 3, 9, 1, 1)
        self.hide_stuff([self.neighbour_counter,self.neighbour_slider,self.neighbour_label])

        self.neighbour_slider.setToolTip('Number of pixels to take into account to do width smoothing')

        # ============= dend width change slider ==================
        self.dend_width_mult_label = QLabel("Dendritic Width Multiplication Factor")
        self.grid.addWidget(self.dend_width_mult_label, 4, 8, 1, 1)
        self.dend_width_mult_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.dend_width_mult_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.dend_width_mult_slider, 4, 2, 1, 6)
        self.dend_width_mult_slider.setMinimum(1)
        self.dend_width_mult_slider.setMaximum(40)
        self.dend_width_mult_slider.setValue(5)
        self.dend_width_mult_slider.singleStep()
        self.dend_width_mult_counter = QLabel(str(self.dend_width_mult_slider.value()))
        self.grid.addWidget(self.dend_width_mult_counter, 4, 9, 1, 1)
        self.hide_stuff([self.dend_width_mult_counter, self.dend_width_mult_slider, self.dend_width_mult_label])

        #============= ML confidence slider ==================
        self.ml_confidence_label = QLabel("ML Confidence")
        self.grid.addWidget(self.ml_confidence_label, 3, 8, 1, 1)
        self.ml_confidence_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.ml_confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.ml_confidence_slider, 3, 2, 1, 6)
        self.ml_confidence_slider.setMinimum(0)
        self.ml_confidence_slider.setMaximum(10)
        self.ml_confidence_slider.setValue(5)
        self.ml_confidence_slider.singleStep()
        self.confidence_counter = QLabel(str(self.ml_confidence_slider.value() / 10))
        self.grid.addWidget(self.confidence_counter, 3, 9, 1, 1)
        self.hide_stuff([self.ml_confidence_label,self.ml_confidence_slider,self.confidence_counter ])

        self.ml_confidence_slider.setToolTip('Select the value to filter out NN suggestions that fall below this confidence')

        #============= spine tolerance slider ==================
        self.tolerance_label = QLabel("Roi Tolerance")
        self.grid.addWidget(self.tolerance_label, 3, 8, 1, 1)
        self.tolerance_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.tolerance_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.tolerance_slider, 3, 2, 1, 6)
        self.tolerance_slider.setMinimum(0)
        self.tolerance_slider.setMaximum(50)
        self.tolerance_slider.setValue(5)
        self.tol_val = 5
        self.tolerance_slider.singleStep()
        self.tolerance_counter = QLabel(str(self.tolerance_slider.value()))
        self.grid.addWidget(self.tolerance_counter, 3, 9, 1, 1)
        self.hide_stuff([self.tolerance_label,self.tolerance_counter,self.tolerance_slider])

        self.tolerance_slider.setToolTip('Select tolerance for ROI generation - higher means larger ROIs')

        #============= spine neck sigma slider ==================
        self.sigma_label = QLabel("Roi Sigma")
        self.grid.addWidget(self.sigma_label, 4, 8, 1, 1)
        self.sigma_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.sigma_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.sigma_slider, 4, 2, 1, 6)
        self.sigma_slider.setMinimum(0)
        self.sigma_slider.setMaximum(20)
        self.sigma_slider.setValue(5)
        self.sigma_val = 5
        self.sigma_slider.singleStep()

        self.sigma_counter = QLabel(str(self.sigma_slider.value()))
        self.grid.addWidget(self.sigma_counter, 4, 9, 1, 1)
        self.hide_stuff([self.sigma_label,self.sigma_counter,self.sigma_slider])

        self.sigma_slider.setToolTip('Select σ value of the canny-edge detection used in the ROI algorithm - lower values means smaller ROIs')

        #============= spine neck sigma slider ==================
        self.spine_neck_sigma_label = QLabel("Neck Width Smoothness")
        self.grid.addWidget(self.spine_neck_sigma_label, 3, 8, 1, 1)
        self.spine_neck_sigma_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.spine_neck_sigma_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.spine_neck_sigma_slider, 3, 2, 1, 6)
        self.spine_neck_sigma_slider.setMinimum(1)
        self.spine_neck_sigma_slider.setMaximum(10)
        self.spine_neck_sigma_slider.setValue(6)
        self.spine_neck_sigma_slider.singleStep()
        self.spine_neck_sigma_counter = QLabel(str(self.spine_neck_sigma_slider.value()))
        self.grid.addWidget(self.spine_neck_sigma_counter, 3, 9, 1, 1)
        self.hide_stuff([self.spine_neck_sigma_counter,self.spine_neck_sigma_slider,self.spine_neck_sigma_label])

        self.spine_neck_sigma_slider.setToolTip('Number of pixels to take into account to do width smoothing')

        # ============= spine neck width change slider ==================
        self.spine_neck_width_mult_label = QLabel("Neck Width Multiplication Factor")
        self.grid.addWidget(self.spine_neck_width_mult_label, 4, 8, 1, 1)
        self.spine_neck_width_mult_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.spine_neck_width_mult_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.spine_neck_width_mult_slider, 4, 2, 1, 6)
        self.spine_neck_width_mult_slider.setMinimum(1)
        self.spine_neck_width_mult_slider.setMaximum(40)
        self.spine_neck_width_mult_slider.setValue(5)
        self.spine_neck_width_mult_slider.singleStep()
        self.spine_neck_width_mult_counter = QLabel(str(self.spine_neck_width_mult_slider.value()))
        self.grid.addWidget(self.spine_neck_width_mult_counter, 4, 9, 1, 1)
        self.hide_stuff([self.spine_neck_width_mult_counter, self.spine_neck_width_mult_slider, self.spine_neck_width_mult_label])

        #============= Dendrite shifting button ==================
        self.Dend_shift_check = QCheckBox(self)
        self.Dend_shift_check.setText("Dendrite shifting")
        self.grid.addWidget(self.Dend_shift_check, 2, 10, 1, 1)
        self.Dend_shift_check.setVisible(False)
        self.Dend_shift = False
        self.Dend_shift_check.stateChanged.connect(lambda state: self.check_changed(state,3))

        self.Dend_shift_check.setToolTip('Check if you only want to use the calculated dendrite to correct for temporal shifting, otherwise uses whole image')

        #============= Local shifting button ==================
        self.local_shift_check = QCheckBox(self)
        self.local_shift_check.setText("Local shifting")
        self.grid.addWidget(self.local_shift_check,3, 10, 1, 1)
        self.local_shift_check.setVisible(False)
        self.local_shift = False
        self.local_shift_check.stateChanged.connect(lambda state: self.check_changed(state,2))
        self.local_shift_check.setToolTip('Check if you want each ROI to be locally motion corrected in addition to the global correction')
        
        # ============= Puncta dendritic threshold slider ==================
        self.puncta_dend_label = QLabel("Threshold dendrite")
        self.grid.addWidget(self.puncta_dend_label,3, 2, 1, 1)
        self.puncta_dend_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.puncta_dend_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.puncta_dend_slider,3 , 3, 1, 9)
        self.puncta_dend_slider.setMinimum(0)
        self.puncta_dend_slider.setMaximum(100)
        self.puncta_dend_slider.setValue(12)
        self.puncta_dend_slider.singleStep()

        self.puncta_dend_counter = QLabel(str(self.puncta_dend_slider.value()))
        self.grid.addWidget(self.puncta_dend_counter, 3, 12, 1, 1)
        self.puncta_dend_slider.setToolTip('Select the detection threshold for the dendritic puncta')

        # ============= Puncta soma threshold slider ==================
        self.puncta_soma_label = QLabel("Threshold synapse/soma")
        self.grid.addWidget(self.puncta_soma_label, 4, 2, 1, 1)
        self.puncta_soma_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.puncta_soma_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.puncta_soma_slider, 4, 3, 1, 9)
        self.puncta_soma_slider.setMinimum(0)
        self.puncta_soma_slider.setMaximum(100)
        self.puncta_soma_slider.setValue(12)
        self.puncta_soma_slider.singleStep()
        self.puncta_soma_slider.setToolTip('Select the synaptic threshold for the dendritic puncta')

        self.puncta_soma_counter = QLabel(str(self.puncta_soma_slider.value()))
        self.grid.addWidget(self.puncta_soma_counter, 4, 12, 1, 1)

        # ============= Puncta sigma range slider ==================

        self.puncta_sigma_label = QLabel("Puncta size")
        self.grid.addWidget(self.puncta_sigma_label,5,2,1,1)
        self.puncta_sigma_range_slider = QLabeledRangeSlider(PyQt5.QtCore.Qt.Horizontal,self)
        self.puncta_sigma_range_slider.setValue((1,2))
        self.grid.addWidget(self.puncta_sigma_range_slider,5,3,1,9)
        self.puncta_sigma_range_slider.setRange(1,10)
        self.puncta_sigma_range_slider.setToolTip('Select the range of puncta sizes ')

        self.hide_stuff([self.puncta_soma_label, self.puncta_soma_counter, self.puncta_soma_slider])
        self.hide_stuff([self.puncta_dend_label, self.puncta_dend_counter, self.puncta_dend_slider])
        self.hide_stuff([self.puncta_sigma_label, self.puncta_sigma_range_slider])

    def set_NN(self):
        """
        Sets the path of a PyTorch neural network (NN) file chosen through a file dialog.
        Updates the UI by displaying the selected file name on a button.
        Sets the model path in `SimVars.model` and enables a specific button (`spine_button_NN`) if another button (`spine_button`) is already enabled.
        Unchecks the "Set NN!" button.
        """
        path = QFileDialog.getOpenFileName(self, "Select pytorch NN!")[0]

        if path:
            self.NN_path = path

            self.button_set_NN.setText("Set NN! ("+os.path.basename(self.NN_path)+")")

            self.SimVars.model = self.NN_path
            self.NN = True
            if(self.spine_button.isEnabled()):
                MakeButtonActive(self.spine_button_NN)

        self.button_set_NN.setChecked(False)

    def download_NN(self):

        self.button_set_NN.setChecked(True)
        QCoreApplication.processEvents()
        self.set_status_message.repaint()
        try:
            load_model(self.SimVars)
            if not os.path.exists(self.default_ML_address):
                os.makedirs(self.default_ML_address)
            torch.save(self.SimVars.model,self.NN_path)

            self.SimVars.model = self.NN_path
            self.button_set_NN.setText("Set NN! (default)")

            self.NN = True
            if(self.spine_button.isEnabled()):
                MakeButtonActive(self.spine_button_NN)

            self.set_status_message.setText('Downloaded and saved in default location')
            QCoreApplication.processEvents()
            self.set_status_message.repaint()
            self.button_set_NN.setChecked(False)
            self.button_set_NN.disconnect()
            self.button_set_NN.clicked.connect(self.set_NN)
        except Exception as e:
            self.set_status_message.setText(e)
            try:
                os.remove('model.pth')
            except:
                pass
            #self.set_status_message.setText('Link was broken - select from computer or cancel')
            self.button_set_NN.disconnect()
            self.button_set_NN.clicked.connect(self.set_NN)
            self.set_NN()



    def spine_tolerance_sigma(self) -> None:
        """
        function that reruns spine roi eval, when slider sigma/tolerance slider is moved
        alters earlier roi polygons
        Returns: None

        """
        self.tol_val = self.tolerance_slider.value()
        self.sigma_val = self.sigma_slider.value()
        points = self.spine_marker.points.astype(int)
        flags = self.spine_marker.flags.astype(int)
        mean = self.SimVars.bgmean[0][0]
        self.set_status_message.setText('Recalculating ROIs')

        # set values in the gui for user
        self.tolerance_counter.setText(str(self.tol_val))
        self.sigma_counter.setText(str(self.sigma_val))

        # remove all old polygons
        for patch in self.mpl.axes.patches:
            patch.remove()

        for lines in self.mpl.axes.lines:
            lines.remove()

        medial_axis_Arr = [Dend.medial_axis.astype(int) for Dend in self.DendArr]

        if(not self.SimVars.multitime_flag):
            tf = self.tiff_Arr[self.actual_timestep,self.actual_channel]
        else:
            tf = self.tiff_Arr[:,self.actual_channel]

        for index, (point,flag) in enumerate(zip(points,flags)):
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                xpert, shift, bgloc,closest_Dend,DendDist,Orientation,_,_ = ROI_And_Neck(
                tf,
                point,
                medial_axis_Arr,
                points,
                mean,
                True,
                sigma=self.sigma_val,
                tol=self.tol_val,
                SpineShift_flag = self.local_shift,
                Mode = 'ROI'
                )
                polygon = np.array(xpert)
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                pol.set_edgecolor('white')
                self.mpl.axes.add_patch(pol)
                self.roi_interactor_list[index].poly = pol
                if(self.roi_interactor_list[index].loc is not None):
                    self.roi_interactor_list[index].points = np.array(pol.xy)-np.array(self.roi_interactor_list[index].loc)
                self.roi_interactor_list[index].line.set_data(pol.xy[:, 0], pol.xy[:, 1])
                OldNeck = self.SpineArr[index].neck
                OldNeckthresh = self.SpineArr[index].neck_thresh
                self.SpineArr[index] = Synapse(list(point),list(bgloc),pts=xpert,shift=shift,
                    channel=self.actual_channel,Syntype=flag,closest_Dend=closest_Dend,DendDist = DendDist*self.SimVars.Unit,Orientation=Orientation,neck = OldNeck,
                    neck_thresh = OldNeckthresh)
            else:
                self.SpineArr[index].points = []
                pt = self.SpineArr[index].location
                tiff_Arr_small = tf[:,
                                max(pt[1] - 50, 0) : min(pt[1] + 50, tf.shape[-2]),
                                max(pt[0] - 50, 0) : min(pt[0] + 50, tf.shape[-1]),
                            ]
                self.SpineArr[index].shift = SpineShift(tiff_Arr_small).T.astype(int).tolist()
                for i in range(self.SimVars.Snapshots):
                    xpert, _, bgloc,closest_Dend,DendDist,Orientation,_,_ = ROI_And_Neck(
                        tf[i],
                        np.array(self.SpineArr[index].location),
                        medial_axis_Arr,
                        points,
                        mean,
                        True,
                        sigma=self.sigma_val,
                        tol=self.tol_val,
                        Mode = 'ROI'
                    )
                    self.SpineArr[index].points.append(xpert)
                    self.SpineArr[index].closest_Dend = closest_Dend
                    self.SpineArr[index].distance_to_Dend = DendDist*self.SimVars.Unit
                    self.SpineArr[index].Orientation = Orientation
                polygon = np.array(self.SpineArr[index].points[self.actual_timestep])
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                pol.set_edgecolor('white')
                self.mpl.axes.add_patch(pol)
                self.roi_interactor_list[index].poly = pol
                self.roi_interactor_list[index].line.set_data(pol.xy[:, 0], pol.xy[:, 1])
            self.set_status_message.setText(self.set_status_message.text()+'.')
            QCoreApplication.processEvents()
            self.set_status_message.repaint()
        
        
        self.SpineArr = SynDistance(self.SpineArr, medial_axis_Arr, self.SimVars.Unit)
        self.mpl.canvas.draw()


    def spine_measure(self):
        """
        function that takes the calculated ROIS and obtains various statistics
        Returns: None
        """

        # I really want to clear the plot and re-plot only the Spine ROIs but 
        # im not sure how to do this with multiple time points
        # I also want the bounding boxes of the spines to be the maximum width of the necks
        # I also want to remove the overlap between spine ROIs and Spine NEcks


        try:
            self.update_plot_handle(
                self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
            )
        except:
            pass

        self.command_box.clear()
        self.show_stuff_coll([])
        for Dend in self.DendArr:
            pol = Polygon(
            Dend.control_points, fill=False, closed=False, animated=False
            )
            self.mpl.axes.add_patch(pol)

        if(hasattr(self,"roi_interactor_bg_list")):
            for S,R in zip(self.SpineArr,self.roi_interactor_bg_list):
                S.bgloc = [R.line.get_data()[0][0],R.line.get_data()[1][0]]
        medial_axis_Arr = [Dend.medial_axis.astype(int) for Dend in self.DendArr]
        for i,(R,L) in enumerate(zip(self.roi_interactor_list,self.line_interactor_list)):
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                try:
                    self.SpineArr[i].points = (R.poly.xy - R.shift[R.Snapshot]).tolist()[:-1]
                    self.SpineArr[i].shift[R.Snapshot] = R.shift[R.Snapshot]
                except:
                    self.SpineArr[i].points = (R.poly.xy).tolist()[:-1]
                if(self.SpineArr[i].type<2):
                    if(self.local_shift):
                        self.SpineArr[i].neck[R.Snapshot] = (L.poly.xy).tolist()
                    else:
                        self.SpineArr[i].neck = (L.poly.xy).tolist()
            else:
                if(self.local_shift):
                    self.SpineArr[i].points[self.actual_timestep] = (R.poly.xy - R.shift[R.Snapshot]).tolist()[:-1]
                else:
                    self.SpineArr[i].points[self.actual_timestep] = (R.poly.xy).tolist()[:-1]
                if(self.SpineArr[i].type<2):
                    
                    self.SpineArr[i].neck[R.Snapshot] = (L.poly.xy).tolist()
                try:
                    self.SpineArr[i].shift[R.Snapshot] = R.shift[R.Snapshot]
                except:
                    pass
            self.SpineArr[i].mean = []
            self.SpineArr[i].min = []
            self.SpineArr[i].max = []
            self.SpineArr[i].RawIntDen = []
            self.SpineArr[i].IntDen = []
            self.SpineArr[i].area = []
            self.SpineArr[i].local_bg = []
            self.SpineArr[i].widths = []
            self.SpineArr[i].neck_length = []

        Measure(self.SpineArr,self.tiff_Arr,self.SimVars,self)

        self.mpl.canvas.draw()
        self.measure_spine_button.setChecked(False)
        MakeButtonActive(self.save_button)
        self.SpinesMeasured = True
        self.show_stuff_coll(["MeasureROI"])
        return None
    
    def dend_measure(self,Dend,i,Dend_Dir):
        """
        function that takes the calculated dendritic segments and saves these
        Returns: None
        """

        Dend_Save_Dir = Dend_Dir + "Dendrite"+str(i)+".npy"
        Dend_Mask_Dir = Dend_Dir + "/Mask_dend"+str(i)+".png"
        if(len(self.SimVars.yLims)>0):
            np.save(Dend_Save_Dir, Dend.control_points -
                np.array([self.SimVars.yLims[0],self.SimVars.xLims[0]]))
        else:
            np.save(Dend_Save_Dir, Dend.control_points)
        try:
            dend_mask = Dend.get_dendritic_surface_matrix() * 255
            try:
                dend_mask = np.pad(dend_mask,((-self.SimVars.xLims[0],self.SimVars.xLims[1]),(-self.SimVars.yLims[0],self.SimVars.Lims[1])),'constant', constant_values=(0,0))
            except:
                pass
            cv.imwrite(Dend_Mask_Dir, dend_mask)
        except Exception as e:
            raise
            return e
        return None

    
    def spine_bg_measure(self):

        """
        function that takes the calculated ROIS and obtains the background statistics
        Returns: None
        """


        self.show_stuff_coll([])
        self.set_status_message.setText("Drag to the optimal background location")
        self.add_commands(["SpineBG_Desc"])

        self.mpl.clear_plot()
        try:
            self.update_plot_handle(
                self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
            )
        except:
            pass
        self.spine_bg_button.setChecked(False)
        self.roi_interactor_bg_list = []
        for index, (S,R) in enumerate(zip(self.SpineArr,self.roi_interactor_list)):
            S.points = (R.poly.xy - R.shift[R.Snapshot])[:-1].tolist()
            ROIline = Line2D(
                np.array(S.points)[:,0],
                np.array(S.points)[:, 1],
                marker=".",
                markerfacecolor="k",
                markersize=10,
                fillstyle="full",
                linestyle="-",
                linewidth=1.5,
                animated=False,
                antialiased=True,
            )
            self.mpl.axes.add_line(ROIline)
            self.roi_interactor_bg_list.append(RoiInteractor_BG(self.mpl.axes,self.mpl.canvas,S))

        self.mpl.canvas.draw()    
        return 0

    def dend_threshold_slider_update(self):

        """
        Updates the puncta dendrite counter and retrieves puncta based on the slider value.

        This method is triggered when the puncta dendrite slider is moved.
        It updates the text of the puncta dendrite counter to reflect the new slider value.
        It then calls the 'get_puncta' method to retrieve puncta based on the updated slider value.

        Returns:
            None
        """
        self.puncta_dend_counter.setText(str(self.puncta_dend_slider.value()))
        self.get_puncta()

    def soma_threshold_slider_update(self):

        """
        Updates the puncta soma counter and retrieves puncta based on the slider value.

        This method is triggered when the puncta soma slider is moved.
        It updates the text of the puncta soma counter to reflect the new slider value.
        It then calls the 'get_puncta' method to retrieve puncta based on the updated slider value.

        Returns:
            None
        """
        self.puncta_soma_counter.setText(str(self.puncta_soma_slider.value()))
        self.get_puncta()

    def puncta_sigma_slider_update(self):
        """
        Updates the puncta soma counter and retrieves puncta based on the slider value.

        This method is triggered when the puncta soma slider is moved.
        It updates the text of the puncta soma counter to reflect the new slider value.
        It then calls the 'get_puncta' method to retrieve puncta based on the updated slider value.

        Returns:
            None
        """
        self.get_puncta()
    def get_puncta(self, max_simga=None):
        """Retrieves and displays puncta based on the current slider values.

        This method shows puncta, adds commands, sets status messages, and performs puncta detection
        using the current slider values. It calculates somatic and dendritic puncta and updates
        the puncta list. Finally, it displays the puncta, updates the status message accordingly,
        and returns None.
        """
        self.show_stuff_coll(["Puncta"])
        self.add_commands(["Puncta"])
        self.set_status_message.setText(self.status_msg["11"])
        QCoreApplication.processEvents()
        self.set_status_message.repaint()
        somas,necks = self.get_soma_polygons()

        soma_thresh = self.puncta_soma_slider.value()/100.0
        dend_thresh = self.puncta_dend_slider.value()/100.0
        sigmas =  self.puncta_sigma_range_slider.value()
        PD = PunctaDetection(self.SimVars,self.tiff_Arr,somas,necks,self.DendArr,dend_thresh,soma_thresh,sigmas)
        somatic_punctas, dendritic_punctas,neck_punctas = PD.GetPunctas()
        self.punctas = [somatic_punctas,dendritic_punctas,neck_punctas]
        self.display_puncta()
        self.measure_puncta_button.setChecked(False)
        self.PunctaCalc = True

        MakeButtonActive(self.save_button)

    def display_puncta(self):
        """Displays the puncta on the plot.

        This method clears the plot and updates it with the current timestep and channel.
        It retrieves the puncta dictionary for the soma and dendrite from the punctas list.
        The puncta for the current timestep and channel are plotted on the plot using different colors.
        The 'soma' puncta are plotted in yellow, and the 'dendrite' puncta are plotted in red.
        The plot is refreshed to reflect the changes.
        """
        self.mpl.clear_plot()
        for i,D in enumerate(self.DendArr):
            D.actual_channel = self.actual_channel
            D.actual_timestep= self.actual_timestep
            dend_surface = D.get_dendritic_surface_matrix()
            if(dend_surface is not None):
                dend_cont = D.get_contours()
                polygon = np.array(dend_cont[0][:, 0, :])
                pol = Polygon(polygon, fill=False, closed=True,color='y')
                self.mpl.axes.add_patch(pol)
        self.mpl.canvas.draw()
        for i,S in enumerate(self.SpineArr):
            polygon = np.array(S.points)
            if(polygon.ndim>2):
                polygon = polygon[0]
            pol = Polygon(polygon, fill=False, closed=True,color='w')
            self.mpl.axes.add_patch(pol)
        self.mpl.canvas.draw()
        try:
            self.update_plot_handle(
                self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
            )
        except:
            pass
        try:
            self.plot_puncta(self.punctas[0][int(self.timestep_slider.value())][int(self.channel_slider.value())],"soma")
            self.plot_puncta(self.punctas[1][int(self.timestep_slider.value())][int(self.channel_slider.value())],"dendrite")
            self.plot_puncta(self.punctas[2][int(self.timestep_slider.value())][int(self.channel_slider.value())],"neck")
        except:
            print("No puncta detected anywhere, try lowering the thresholds")
            pass

    def plot_puncta(self,puncta_dict,flag='dendrite'):

        """Plots the puncta on the plot.

        This method takes a puncta dictionary and a flag indicating the type of puncta ('soma' or 'dendrite').
        It iterates over the puncta in the dictionary and plots each punctum as a circle on the plot.
        The circle's center coordinates, radius, and color are determined based on the punctum type.
        The plotted circles are added as patches to the plot.
        The plot is refreshed to reflect the changes.
        """
        for p in puncta_dict:
            puncta_x,puncta_y = p.location
            puncta_r          = p.radius
            if(flag=='dendrite'):
                c = plt.Circle((puncta_x, puncta_y), puncta_r, color="red", linewidth=2, fill=False)
            elif flag == 'soma':
                c = plt.Circle((puncta_x, puncta_y), puncta_r, color="y", linewidth=2, fill=False)
            elif flag == 'neck':
                c = plt.Circle((puncta_x, puncta_y), puncta_r, color="g", linewidth=2, fill=False)
            self.mpl.axes.add_patch(c)
        QCoreApplication.processEvents()
        self.mpl.canvas.draw()


    def get_soma_polygons(self):

        """Returns a dictionary of soma polygons.

        This method retrieves the spine array and checks for spines of type 2, which represent soma polygons.
        If the simulation mode is 'Luminosity', the points of the spine array are added as an array to the soma dictionary.
        Otherwise, only the first point of the spine array is added as an array to the soma dictionary.
        The soma dictionary maps a unique identifier to each soma polygon array.
        The resulting soma dictionary is returned.
        """
        soma_count = 0
        soma_dict = []
        neck_dict = []
        self.SaveROIstoSpine()
        for S in self.SpineArr:
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                soma_dict.append(np.asarray(S.points))
                neck_dict.append(np.asarray(S.neck_contours))
            else:
                soma_dict.append(np.asarray(S.points[0]))
                if(len(S.neck_contours)>0):
                    neck_dict.append(np.asarray(S.neck_contours[0]))
                else:
                    neck_dict.append(np.asarray(S.neck_contours))


        return soma_dict,neck_dict

    def save_results(self):
        """Save the results of the evaluation.

        The method saves various simulation results, such as background data, dendrite measurements,
        spine masks, and synaptic dictionaries. It also updates the status message and redraws the canvas.

        Returns:
            None
        """

        self.add_commands([])
        self.show_stuff_coll([])
        SaveFlag = np.array([True,True,True])
        np.save(self.SimVars.Dir + "background.npy",self.SimVars.bgmean)
        if(len(self.DendArr)>0):
            Dend_Dir = self.SimVars.Dir + "Dendrite/"
            try:
                if os.path.exists(Dend_Dir):
                    shutil.copytree(Dend_Dir, Dend_Dir[:-1]+'temp/')
                    shutil.rmtree(Dend_Dir)
                os.mkdir(path=Dend_Dir)
                for i,Dend in enumerate(self.DendArr):
                    self.dend_measure(Dend,i,Dend_Dir)
                try:
                    DendSave_csv(Dend_Dir,self.DendArr,-np.array([self.SimVars.yLims[0],self.SimVars.xLims[0]]))
                    DendSave_json(Dend_Dir,self.DendArr,self.tiff_Arr,self.SimVars.Snapshots,self.SimVars.Channels,self.SimVars.Unit,-np.array([self.SimVars.yLims[0],self.SimVars.xLims[0]]))
                    DendSave_imj(Dend_Dir,self.DendArr,-np.array([self.SimVars.yLims[0],self.SimVars.xLims[0]]))
                except:
                    DendSave_csv(Dend_Dir,self.DendArr,-np.array([0,0]))
                    DendSave_json(Dend_Dir,self.DendArr,self.tiff_Arr,self.SimVars.Snapshots,self.SimVars.Channels,self.SimVars.Unit,-np.array([0,0]))
                    DendSave_imj(Dend_Dir,self.DendArr,-np.array([0,0]))
                if os.path.exists(Dend_Dir[:-1]+'temp/'):
                    shutil.rmtree(Dend_Dir[:-1]+'temp/')
            except Exception as e:
                print(e)
                if os.path.exists(Dend_Dir[:-1]+'temp/'):
                    shutil.rmtree(Dend_Dir)
                    shutil.copytree(Dend_Dir[:-1]+'temp/', Dend_Dir)
                    shutil.rmtree(Dend_Dir[:-1]+'temp/')
                if DevMode: print(e)
                SaveFlag[0] = False
                pass
        else:
            SaveFlag[0] = False

        if(len(self.SpineArr)>0):
            self.SaveROIstoSpine()
            if(not self.SpinesMeasured):
                self.spine_measure()
                self.SpinesMeasured = True
            T = np.argsort([s.distance for s in self.SpineArr])
            try:
                Spine_Dir = self.SimVars.Dir + "Spine/"
                if os.path.exists(Spine_Dir):
                    shutil.copytree(Spine_Dir, Spine_Dir[:-1]+'temp/')
                    for file_name in os.listdir(Spine_Dir):
                        file_path = os.path.join(Spine_Dir, file_name)
                        
                        # check if the file is the one to keep
                        if ((file_name.startswith('Synapse_a') and self.SimVars.Mode=="Luminosity")
                            or (file_name.startswith('Synapse_l') and self.SimVars.Mode=="Area")):
                            continue  # skip the file if it's the one to keep
                        # delete the file if it's not the one to keep
                        try:
                            os.remove(file_path)
                        except:
                            shutil.rmtree(file_path)
                else:
                    os.mkdir(path=Spine_Dir)

                orderedSpineArr = [self.SpineArr[t] for t in T]
                #save spine masks
                for i,t in  enumerate(T):
                    if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                        R = self.roi_interactor_list[t]
                        Spine_Mask_Dir = Spine_Dir + "Mask_" + str(i) + ".png"
                        xperts = R.getPolyXYs()
                        mask = np.zeros_like(self.tiff_Arr[0,0])

                        c = np.clip(xperts[:, 0],0,self.tiff_Arr.shape[-1]-1)
                        r = np.clip(xperts[:, 1],0,self.tiff_Arr.shape[-2]-1)
                        rr, cc = polygon(r, c)
                        mask[rr, cc] = 255
                        try:
                            mask = np.pad(mask,((-self.SimVars.xLims[0],self.SimVars.xLims[1]),(-self.SimVars.yLims[0],self.SimVars.yLims[1])),'constant', constant_values=(0,0))
                        except:
                            pass
                        cv.imwrite(Spine_Mask_Dir, mask)
                    else:
                        for j,xperts in enumerate(self.SpineArr[t].points):
                            xperts = np.array(xperts)
                            Spine_Mask_Dir = Spine_Dir + "Mask_" + str(i) +"_t"+str(j)+ ".png"
                            mask = np.zeros_like(self.tiff_Arr[0,0])

                            c = np.clip(xperts[:, 0],0,self.tiff_Arr.shape[-1]-1)
                            r = np.clip(xperts[:, 1],0,self.tiff_Arr.shape[-2]-1)
                            rr, cc = polygon(r, c)
                            mask[rr, cc] = 255
                            try:
                                mask = np.pad(mask,((-self.SimVars.xLims[0],self.SimVars.xLims[1]),(-self.SimVars.yLims[0],self.SimVars.yLims[1])),'constant', constant_values=(0,0))
                            except:
                                    pass
                            cv.imwrite(Spine_Mask_Dir, mask)
                nSnaps = self.number_timesteps if self.SimVars.multitime_flag else 1
                nChans = self.number_channels if self.SimVars.multiwindow_flag else 1
                try:
                    SaveSynDict(orderedSpineArr, Spine_Dir, self.SimVars.Mode,[self.SimVars.yLims,self.SimVars.xLims])
                    SpineSave_csv(Spine_Dir,orderedSpineArr,nChans,nSnaps,self.SimVars.Mode,[self.SimVars.yLims,self.SimVars.xLims],self.local_shift)
                    SpineSave_imj(Spine_Dir,orderedSpineArr)
                except:
                    if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                        SaveSynDict(orderedSpineArr, Spine_Dir, "Luminosity",[[],[]])
                        SpineSave_csv(Spine_Dir,orderedSpineArr,nChans,nSnaps,"Luminosity",[[],[]],self.local_shift)
                        SpineSave_imj(Spine_Dir,orderedSpineArr)
                    else:
                        SaveSynDict(orderedSpineArr, Spine_Dir, self.SimVars.Mode,[[],[]])
                        SpineSave_csv(Spine_Dir,orderedSpineArr,nChans,nSnaps,"Luminosity",[[],[]],self.local_shift)
                        SpineSave_imj(Spine_Dir,orderedSpineArr)
                self.PlotSyn()
                if os.path.exists(Spine_Dir[:-1]+'temp/'):
                    shutil.rmtree(Spine_Dir[:-1]+'temp/')
            except Exception as e:
                print(e)
                if os.path.exists(Spine_Dir[:-1]+'temp/'):
                   shutil.rmtree(Spine_Dir)
                   shutil.copytree(Spine_Dir[:-1]+'temp/', Spine_Dir)
                   shutil.rmtree(Spine_Dir[:-1]+'temp/')
                if DevMode: print(e)
                SaveFlag[1] = False
                pass
        else:
            SaveFlag[1] = False
        if(len(self.punctas)>0):
            for sp in self.punctas[0]:
                for p1 in sp:
                    for p2 in p1:  
                        p2.RoiID =  T[p2.RoiID]
                        p2.distance = self.SpineArr[p2.RoiID].distance
            try:
                puncta_Dir = self.SimVars.Dir + "/Puncta/"
                if os.path.exists(puncta_Dir):
                    shutil.rmtree(puncta_Dir)
                os.makedirs(puncta_Dir, exist_ok=True)

                save_puncta(puncta_Dir,self.punctas,[self.SimVars.yLims,self.SimVars.xLims])
            except Exception as e:
                if DevMode: print(e)
                SaveFlag[2] = False
                pass
        else:
            SaveFlag[2] = False
        self.SaveSettings()
        if(SaveFlag.all()):
            self.set_status_message.setText(self.status_msg["7"])
        elif(not SaveFlag.any()):
            Text = "Something went wrong! You haven't saved anything - please retry."
        else:
            Text = ""
            if(SaveFlag[0]):
                Text += "Dendrite saved properly, "
            if(SaveFlag[1]):
                Text += "Synapses saved properly, "
            if(SaveFlag[2]):
                Text += "Puncta saved properly"
            self.set_status_message.setText(Text)
        self.save_button.setChecked(False)
        self.mpl.canvas.draw()

    def SaveSettings(self):
        """
        Saves the current settings to a file.

        The settings include multi-time flag, resolution, image threshold, dendritic width,
        ML confidence, ROI tolerance, and ROI sigma.

        Returns:
            None

        """
        Global_Settings_File = os.path.expanduser("~") + "/SpyDenSettings.txt"
        Settings_File = self.SimVars.Dir + "Settings.txt"


        if os.path.exists(Global_Settings_File):
            os.remove(Global_Settings_File)

        file = open(Global_Settings_File, "w")
        values = [("multi-time",self.SimVars.multitime_flag),
                  ("resolution",self.SimVars.Unit),
                  ("Analysis mode",self.SimVars.Mode),
                  ("Image threshold",self.thresh_slider.value()),
                  ("Dendritic width",self.neighbour_slider.value()),
                  ("Dend. width multiplier",self.dend_width_mult_slider.value()),
                  ("ML Confidence",self.ml_confidence_slider.value()),
                  ("ROI Tolerance",self.tolerance_slider.value()),
                  ("ROI Sigma",self.sigma_slider.value()),
                  ("ROI Sigma",self.sigma_slider.value()),
                  ("ROI Sigma",self.sigma_slider.value()),
                  ("Dendritic puncta threshold",self.puncta_dend_slider.value()),
                  ("Somatic puncta threshold",self.puncta_soma_slider.value()),
                  ("MLLocation",self.NN_path),
                  ("Dend. shift",self.Dend_shift_check.isChecked()),
                  ("Puncta sigma range","-".join([str(x) for x in self.puncta_sigma_range_slider.value()]))
                  ]
        for value in values:
            file.write(value[0]+":"+str(value[1]) + "\n")
        file.close()

        if os.path.exists(Settings_File):
            os.remove(Settings_File)

        file = open(Settings_File, "w")
        values = [("multi-time",self.SimVars.multitime_flag),
                  ("resolution",self.SimVars.Unit),
                  ("Analysis mode",self.SimVars.Mode),
                  ("Image threshold",self.thresh_slider.value()),
                  ("Dendritic width",self.neighbour_slider.value()),
                  ("Dend. width multiplier",self.dend_width_mult_slider.value()),
                  ("ML Confidence",self.ml_confidence_slider.value()),
                  ("ROI Tolerance",self.tolerance_slider.value()),
                  ("ROI Sigma",self.sigma_slider.value()),
                  ("Neck Sigma",self.spine_neck_sigma_slider.value()),
                  ("Neck width multiplier",self.spine_neck_width_mult_slider.value()),
                  ("Dendritic puncta threshold",self.puncta_dend_slider.value()),
                  ("Somatic puncta threshold",self.puncta_soma_slider.value()),
                  ("MLLocation",self.NN_path),
                  ("Dend. shift",self.Dend_shift_check.isChecked()),
                  ("Puncta sigma range","-".join([str(x) for x in self.puncta_sigma_range_slider.value()]))
                  ]
        for value in values:
            file.write(value[0]+":"+str(value[1]) + "\n")
        file.close()
    def PlotSyn(self):

        """
        Input:
                
                tiff_Arr (np.array) : The pixel values of the of tiff files
                SynArr (np.array of Synapses) : Array holding synaps information
                SimVars  (class)    : The class holding all simulation parameters

        Output:
                N/A
        Function:
                Plots Stuff
        """ 
        if(self.SimVars.Mode=="Area"):
            for i,t in enumerate(self.tiff_Arr[:,0]):
                fig = plt.figure()

                plt.imshow(self.tiff_Arr[0,0])
                if(hasattr(self,'roi_interactor_list')):
                    T = np.argsort([s.distance for s in self.SpineArr])
                    for j,t in enumerate(T):
                        S = self.SpineArr[t]
                        xy = np.array(S.points[i])
                        plt.plot(xy[:,0],xy[:,1],'-r')

                        labelpt = np.array(S.location)
                        plt.text(labelpt[0] ,labelpt[1], str(j), color='y')
                try:
                    for j,D in enumerate(self.DendArr):
                        plt.plot(D.control_points[:,0],D.control_points[:,1],'-k')
                        labelpt = D.control_points[1,:]
                        plt.text(labelpt[0] ,labelpt[1], str(j), color='k')
                except Exception as e:
                    print(e)
                plt.tight_layout()

                fig.savefig(self.SimVars.Dir+'Spine/ROIs_'+str(i)+'.png')

        else:
            fig = plt.figure()
            plt.imshow(self.tiff_Arr[0,0])
            if(hasattr(self,'roi_interactor_list')):
                T = np.argsort([s.distance for s in self.SpineArr])
                for i,t in enumerate(T):
                    S = self.SpineArr[t]
                    xy = np.array(S.points)
                    plt.plot(xy[:,0],xy[:,1],'-r')

                    labelpt = np.array(S.location)
                    plt.text(labelpt[0] ,labelpt[1], str(i), color='y')
            try:
                for i,D in enumerate(self.DendArr):
                    plt.plot(D.control_points[:,0],D.control_points[:,1],'-k')
                    labelpt = D.control_points[1,:]
                    plt.text(labelpt[0] ,labelpt[1], str(i), color='k')
            except Exception as e:
                print(e)
            plt.tight_layout()

            fig.savefig(self.SimVars.Dir+'Spine/ROIs.png')

    def spine_ROI_eval(self):

        """Evaluate and calculate ROIs for spines.

        The method performs the evaluation and calculation of regions of interest (ROIs) for spines.
        It sets up the necessary configurations, clears previous ROIs and plot elements, and then
        iterates through the specified points and flags to find the shapes and create ROIs accordingly.
        The resulting ROIs are displayed on the plot and stored in the `SpineArr` list.

        Returns:
            None
        """
        self.PunctaCalc = False
        self.spine_marker.disconnect()
        self.show_stuff_coll(["SpineROI"])
        self.set_status_message.setText('Calculating ROIs')
        self.sigma_slider.setValue(self.sigma_val)
        self.tolerance_slider.setValue(self.tol_val)

        self.add_commands(["SpineROI_Desc","SpineROI_Func"])
        if(hasattr(self,"roi_interactor_list")):
            for R in self.roi_interactor_list:
                R.clear()
        if(hasattr(self,"line_interactor_list")):
            for L in self.line_interactor_list:
                L.clear()
        for patch in self.mpl.axes.patches:
            patch.remove()

        for lines in self.mpl.axes.lines:
            lines.remove()

        self.mpl.clear_plot()
        try:
            self.update_plot_handle(
                self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
            )
        except:
            pass
        points = self.spine_marker.points.astype(int)
        flags  = self.spine_marker.flags.astype(int)
        mean = self.SimVars.bgmean[0][0]

        medial_axis_Arr = [Dend.medial_axis.astype(int) for Dend in self.DendArr]

        if(not self.SimVars.multitime_flag):
            tf = self.tiff_Arr[self.actual_timestep,self.actual_channel]
        else:
            tf = self.tiff_Arr[:,self.actual_channel]


        self.SpineArr = []
        self.roi_interactor_list = []
        self.line_interactor_list = []
        for index, (point,flag) in enumerate(zip(points,flags)):
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                xpert, shift, bgloc,closest_Dend,DendDist,Orientation,neck,neck_thresh = ROI_And_Neck(
                    tf,
                    point,
                    medial_axis_Arr,
                    points,
                    mean,
                    True,
                    sigma=self.sigma_val,
                    tol  = self.tol_val,
                    SpineShift_flag = self.local_shift
                )

                polygon = np.array(xpert)
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                if(self.local_shift and self.SimVars.multitime_flag):
                    neck_arr = []
                    neck_t_arr = []
                    for i in range(self.SimVars.Snapshots):
                        _, _, _,_,_,_,neck,neck_thresh = ROI_And_Neck(
                            tf[i],
                            point+ np.array([shift[i][0],shift[i][1]]),
                            medial_axis_Arr,
                            points,
                            mean,
                            True,
                            sigma=self.sigma_val,
                            tol  = self.tol_val,
                            SpineShift_flag = False,
                            Mode = 'Neck'
                        )
                        if(self.DendArr[closest_Dend].get_contours() is not None):
                            cp = Path(self.DendArr[closest_Dend].get_contours()[0].squeeze())
                            inside = cp.contains_points(neck)
                            if(inside.any()):
                                crossing_index = np.where(inside)[0][0]
                                neck = neck[:crossing_index]
                        neck_arr.append(neck)
                        neck_t_arr.append(neck_thresh)
                    if(flag < 2):
                        pol_line = Polygon(neck_arr[self.actual_timestep], fill=False, closed=False, animated=True)
                        self.mpl.axes.add_patch(pol_line)
                        self.line_interactor_list.append(LineInteractor(self.mpl.axes, self.mpl.canvas, pol_line,True,markerprops=['k','g',1.2]))
                    else:
                        self.line_interactor_list.append([])
                else:
                    if(self.DendArr[closest_Dend].get_contours() is not None):
                        cp = Path(self.DendArr[closest_Dend].get_contours()[0].squeeze())
                        inside = cp.contains_points(neck)
                        if(inside.any()):
                            crossing_index = np.where(inside)[0][0]
                            neck = neck[:crossing_index]
                    neck_arr = neck
                    neck_t_arr = neck_thresh
                    if(flag < 2):
                        pol_line = Polygon(neck_arr, fill=False, closed=False, animated=True)
                        self.mpl.axes.add_patch(pol_line)
                        self.line_interactor_list.append(LineInteractor(self.mpl.axes, self.mpl.canvas, pol_line,True,markerprops=['k','g',1.2]))
                    else:
                        self.line_interactor_list.append([])
                if(not self.local_shift):
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas,pol,loc=point))
                else:
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, 
                        pol,point,shift,self.actual_timestep,self.SimVars.Snapshots))
                if(flag < 2):
                    self.SpineArr.append(Synapse(list(point),list(bgloc),pts=xpert,
                        shift=shift,channel=self.actual_channel,Syntype=flag,closest_Dend=closest_Dend,
                        DendDist = DendDist*self.SimVars.Unit,Orientation = Orientation,neck = neck_arr,neck_thresh = neck_t_arr))
                else:
                    self.SpineArr.append(Synapse(list(point),list(bgloc),pts=xpert,
                        shift=shift,channel=self.actual_channel,Syntype=flag,closest_Dend=closest_Dend,
                        DendDist = DendDist*self.SimVars.Unit,Orientation = Orientation,neck = list(point)))
                
            else:
                self.SpineArr.append(Synapse(list(point),[],pts=[],shift=[],channel=self.actual_channel,Syntype=flag))
                
                tiff_Arr_small = tf[:,
                                max(point[1] - 50, 0) : min(point[1] + 50, tf.shape[-2]),
                                max(point[0] - 50, 0) : min(point[0] + 50, tf.shape[-1]),
                            ]
                self.SpineArr[-1].shift = SpineShift(tiff_Arr_small).T.astype(int).tolist()
                neck_arr = []
                neck_t_arr = []
                for i in range(self.SimVars.Snapshots):

                    xpert, shift, radloc,closest_Dend,x,Orientation,_,_ = ROI_And_Neck(
                        tf[i],
                        np.array(self.SpineArr[-1].location),
                        medial_axis_Arr,
                        points,
                        mean,
                        True,
                        sigma=self.sigma_val,
                        tol  = self.tol_val,
                        SpineShift_flag = self.local_shift,
                        Mode='ROI'
                    )

                    self.SpineArr[-1].points.append(xpert)
                    self.SpineArr[-1].closest_Dend = closest_Dend
                    self.SpineArr[-1].Orientation  = Orientation
                    self.SpineArr[-1].distance_to_Dend = x*self.SimVars.Unit
                    
                    if(self.local_shift):
                        _, _, _,_,_,_,neck,neck_thresh = ROI_And_Neck(
                            tf[i],
                            np.array(self.SpineArr[-1].location)+ np.array([self.SpineArr[-1].shift[i][0],self.SpineArr[-1].shift[i][1]]),
                            medial_axis_Arr,
                            points,
                            mean,
                            True,
                            sigma=self.sigma_val,
                            tol  = self.tol_val,
                            SpineShift_flag = False,
                            Mode = 'Neck'
                        )
                    else:
                        _, _, _,_,_,_,neck,neck_thresh = ROI_And_Neck(
                            tf[i],
                            np.array(self.SpineArr[-1].location),
                            medial_axis_Arr,
                            points,
                            mean,
                            True,
                            sigma=self.sigma_val,
                            tol  = self.tol_val,
                            SpineShift_flag = False,
                            Mode = 'Neck'
                        )
                    if(self.DendArr[closest_Dend].get_contours() is not None):
                        cp = Path(self.DendArr[closest_Dend].get_contours()[0].squeeze())
                        inside = cp.contains_points(neck)
                        if(inside.any()):
                            crossing_index = np.where(inside)[0][0]
                            neck = neck[:crossing_index]
                    neck_arr.append(neck)
                    neck_t_arr.append(neck_thresh)
                if(flag < 2):
                    pol_line = Polygon(neck_arr[self.actual_timestep], fill=False, closed=False, animated=True)
                    self.mpl.axes.add_patch(pol_line)
                    self.line_interactor_list.append(LineInteractor(self.mpl.axes, self.mpl.canvas, pol_line,True,markerprops=['k','g',1.2]))
                    self.SpineArr[-1].neck  = neck_arr
                    self.SpineArr[-1].neck_thresh  = neck_t_arr
                else:
                    self.SpineArr[-1].neck = list(self.SpineArr[-1].location)
                    self.line_interactor_list.append([])
                polygon = np.array(self.SpineArr[-1].points[self.actual_timestep])
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                if(not self.local_shift):
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas,pol,loc=point))
                else:
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, 
                        pol,point,self.SpineArr[-1].shift,self.actual_timestep,self.SimVars.Snapshots))

            self.set_status_message.setText(self.set_status_message.text()+'.')
            QCoreApplication.processEvents()
            self.set_status_message.repaint()

        if(self.SimVars.Mode=="Luminosity"): MakeButtonActive(self.spine_bg_button)
        MakeButtonActive(self.measure_spine_button)
        MakeButtonActive(self.measure_puncta_button)
        MakeButtonActive(self.save_button)
        self.spine_button_ROI.setChecked(False)

        self.SpineArr = SynDistance(self.SpineArr, medial_axis_Arr, self.SimVars.Unit)

        self.set_status_message.setText(self.status_msg["9"])

        self.SpinesMeasured = False

    def old_ROI_eval(self):

        """Evaluate and calculate ROIs for spines using the old files.

        The method performs the evaluation and calculation of regions of interest (ROIs) for spines
        using the old files. It clears previous ROIs and plot elements, and then iterates through
        the specified points and flags to find the shapes and create ROIs accordingly. The resulting
        ROIs are displayed on the plot and stored in the `SpineArr` list.

        Returns:
            None
        """

        self.PunctaCalc = False
        self.spine_marker.disconnect()
        self.SpineArr = np.array(self.SpineArr)[[sp.location in self.spine_marker.points.astype(int).tolist() for sp in self.SpineArr]].tolist()
        self.show_stuff_coll(["SpineROI"])
        if(hasattr(self,"roi_interactor_list")):
            for R in self.roi_interactor_list:
                R.clear()
        if(hasattr(self,"line_interactor_list")):
            for L in self.line_interactor_list:
                L.clear()
        for patch in self.mpl.axes.patches:
            patch.remove()
        for lines in self.mpl.axes.lines:
            lines.remove()

        self.mpl.clear_plot()
        try:
            self.update_plot_handle(
                self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
            )
        except:
            pass
        mean = self.SimVars.bgmean[0][0]

        medial_axis_Arr = [Dend.medial_axis.astype(int) for Dend in self.DendArr]

        self.roi_interactor_list = []
        self.line_interactor_list = []
        for S in self.SpineArr:
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                xpert = S.points
                polygon = np.array(xpert)
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                if(S.shift is None):
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas,pol,loc=S.location))
                else:
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, 
                        pol,S.location,S.shift,self.actual_timestep,self.SimVars.Snapshots))
                    self.local_shift = True
                    self.local_shift_check.blockSignals(True)
                    self.local_shift_check.setChecked(True)
                    self.local_shift_check.blockSignals(False)
                if(S.type < 2):
                    try:
                        pol_line = Polygon(S.neck[self.actual_timestep], fill=False, closed=False, animated=True)
                        self.mpl.axes.add_patch(pol_line)
                        self.line_interactor_list.append(LineInteractor(self.mpl.axes, self.mpl.canvas, pol_line,True,markerprops=['k','g',1.2]))
                    except:
                        pol_line = Polygon(S.neck, fill=False, closed=False, animated=True)
                        self.mpl.axes.add_patch(pol_line)
                        self.line_interactor_list.append(LineInteractor(self.mpl.axes, self.mpl.canvas, pol_line,True,markerprops=['k','g',1.2]))
                else:
                    self.line_interactor_list.append([])

            else:
                polygon = np.array(S.points[self.actual_timestep])
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                if(S.shift is None):
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas,pol,loc=S.location))
                else:
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, 
                        pol,S.location,S.shift,self.actual_timestep,self.SimVars.Snapshots))
                    self.local_shift = True
                    self.local_shift_check.blockSignals(True)
                    self.local_shift_check.setChecked(True)
                    self.local_shift_check.blockSignals(False)
                if(S.type < 2):
                    pol_line = Polygon(S.neck[self.actual_timestep], fill=False, closed=False, animated=True)
                    self.mpl.axes.add_patch(pol_line)
                    self.line_interactor_list.append(LineInteractor(self.mpl.axes, self.mpl.canvas, pol_line,True,markerprops=['k','g',1.2]))
                else:
                    self.line_interactor_list.append([])
 
        points = self.spine_marker.points.astype(int)[[list(sp) not in [S.location for S in self.SpineArr] for sp in self.spine_marker.points]]
        flags  = self.spine_marker.flags.astype(int)[[list(sp) not in [S.location for S in self.SpineArr] for sp in self.spine_marker.points]]

        if(not self.SimVars.multitime_flag):
            tf = self.tiff_Arr[self.actual_timestep,self.actual_channel]
        else:
            tf = self.tiff_Arr[:,self.actual_channel]

        for index, (point,flag) in enumerate(zip(points,flags)):
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                xpert, shift, bgloc,closest_Dend,DendDist,Orientation,neck,neck_thresh = ROI_And_Neck(
                    tf,
                    point,
                    medial_axis_Arr,
                    points,
                    mean,
                    True,
                    sigma=self.sigma_val,
                    tol  = self.tol_val,
                    SpineShift_flag=self.local_shift,
                )


                polygon = np.array(xpert)
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                if(not self.local_shift):
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas,pol,loc=point))
                else:
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, 
                        pol,point,shift,self.actual_timestep,self.SimVars.Snapshots))

                if(self.local_shift and self.SimVars.multitime_flag):
                    neck_arr = []
                    neck_t_arr = []
                    for i in range(self.SimVars.Snapshots):
                        _, _, _,_,_,_,neck,neck_thresh = ROI_And_Neck(
                            tf[i],
                            point+ np.array([shift[i][0],shift[i][1]]),
                            medial_axis_Arr,
                            points,
                            mean,
                            True,
                            sigma=self.sigma_val,
                            tol  = self.tol_val,
                            SpineShift_flag = False,
                            Mode = 'Neck'
                        )
                        if(self.DendArr[closest_Dend].get_contours() is not None):
                            cp = Path(self.DendArr[closest_Dend].get_contours()[0].squeeze())
                            inside = cp.contains_points(neck)
                            if(inside.any()):
                                crossing_index = np.where(inside)[0][0]
                                neck = neck[:crossing_index]
                        neck_arr.append(neck)
                        neck_t_arr.append(neck_thresh)
                    if(flag < 2):
                        pol_line = Polygon(neck_arr[self.actual_timestep], fill=False, closed=False, animated=True)
                        self.mpl.axes.add_patch(pol_line)
                        self.line_interactor_list.append(LineInteractor(self.mpl.axes, self.mpl.canvas, pol_line,True,markerprops=['k','g',1.2]))
                    else:
                        self.line_interactor_list.append([])
                else:
                    if(self.DendArr[closest_Dend].get_contours() is not None):
                        cp = Path(self.DendArr[closest_Dend].get_contours()[0].squeeze())
                        inside = cp.contains_points(neck)
                        if(inside.any()):
                            crossing_index = np.where(inside)[0][0]
                            neck = neck[:crossing_index]
                    neck_arr = neck
                    neck_t_arr = neck_thresh
                    if(flag < 2):
                        pol_line = Polygon(neck_arr, fill=False, closed=False, animated=True)
                        self.mpl.axes.add_patch(pol_line)
                        self.line_interactor_list.append(LineInteractor(self.mpl.axes, self.mpl.canvas, pol_line,True,markerprops=['k','g',1.2]))
                    else:
                        self.line_interactor_list.append([])

                if(flag < 2):
                    self.SpineArr.append(Synapse(list(point),list(bgloc),pts=xpert,
                        shift=shift,channel=self.actual_channel,Syntype=flag,closest_Dend=closest_Dend,
                        DendDist = DendDist*self.SimVars.Unit,Orientation = Orientation,neck = neck_arr,neck_thresh = neck_t_arr))
                else:
                    self.SpineArr.append(Synapse(list(point),list(bgloc),pts=xpert,
                        shift=shift,channel=self.actual_channel,Syntype=flag,closest_Dend=closest_Dend,
                        DendDist = DendDist*self.SimVars.Unit,Orientation = Orientation,neck = list(point)))

            else:
                self.SpineArr.append(Synapse(list(point),[],pts=[],shift=[],channel=self.actual_channel,Syntype=flag))
                
                tiff_Arr_small = tf[:,
                                max(point[1] - 50, 0) : min(point[1] + 50, tf.shape[-2]),
                                max(point[0] - 50, 0) : min(point[0] + 50, tf.shape[-1]),
                            ]
                self.SpineArr[-1].shift = SpineShift(tiff_Arr_small).T.astype(int).tolist()
                for i in range(self.SimVars.Snapshots):
                    xpert, _, radloc,closest_Dend,x,Orientation,neck,neck_thresh = ROI_And_Neck(
                        tf[i],
                        np.array(self.SpineArr[-1].location),
                        medial_axis_Arr,
                        points,
                        mean,
                        True,
                        sigma=self.sigma_val,
                        tol  = self.tol_val,
                    )
                    if(i==0):
                        self.SpineArr[-1].distance_to_Dend = x*self.SimVars.Unit
                        self.SpineArr[-1].closest_Dend = closest_Dend
                        self.SpineArr[-1].Orientation  = Orientation
                    self.SpineArr[-1].points.append(xpert)

                    if(self.local_shift):
                        _, _, _,_,_,_,neck,neck_thresh = ROI_And_Neck(
                            tf[i],
                            np.array(self.SpineArr[-1].location)+ np.array([self.SpineArr[-1].shift[i][0],self.SpineArr[-1].shift[i][1]]),
                            medial_axis_Arr,
                            points,
                            mean,
                            True,
                            sigma=self.sigma_val,
                            tol  = self.tol_val,
                            SpineShift_flag = False,
                            Mode = 'Neck'
                        )
                    else:
                        _, _, _,_,_,_,neck,neck_thresh = ROI_And_Neck(
                            tf[i],
                            np.array(self.SpineArr[-1].location),
                            medial_axis_Arr,
                            points,
                            mean,
                            True,
                            sigma=self.sigma_val,
                            tol  = self.tol_val,
                            SpineShift_flag = False,
                            Mode = 'Neck'
                        )
                    if(self.DendArr[closest_Dend].get_contours() is not None):
                        cp = Path(self.DendArr[closest_Dend].get_contours()[0].squeeze())
                        inside = cp.contains_points(neck)
                        if(inside.any()):
                            crossing_index = np.where(inside)[0][0]
                            neck = neck[:crossing_index]
                    if flag < 2:
                        self.SpineArr[-1].neck  = neck_arr
                        self.SpineArr[-1].neck_thresh  = neck_t_arr
                    else:
                        self.SpineArr[-1] = list(self.SpineArr[-1].location)
                polygon = np.array(self.SpineArr[-1].points[self.actual_timestep])
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                if(not self.local_shift):
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas,pol,loc=point))
                else:
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, 
                        pol,point,self.SpineArr[-1].shift,self.actual_timestep,self.SimVars.Snapshots))
                if(flag < 2):
                    pol_line = Polygon(neck_arr[self.actual_timestep], fill=False, closed=False, animated=True)
                    self.mpl.axes.add_patch(pol_line)
                    self.line_interactor_list.append(LineInteractor(self.mpl.axes, self.mpl.canvas, pol_line,True,markerprops=['k','g',1.2]))
                else:
                    self.line_interactor_list.append([])


            self.set_status_message.setText(self.set_status_message.text()+'.')
            QCoreApplication.processEvents()
            self.set_status_message.repaint()
        if(self.SimVars.Mode=="Luminosity"): MakeButtonActive(self.spine_bg_button)
        MakeButtonActive(self.measure_spine_button)
        MakeButtonActive(self.measure_puncta_button)
        MakeButtonActive(self.save_button)
        self.old_ROI_button.setChecked(False)

        self.set_status_message.setText(self.status_msg["9"])

        self.SpineArr = SynDistance(self.SpineArr, medial_axis_Arr, self.SimVars.Unit)

        self.SpinesMeasured = False

    def clear_settings(self):
        self.LoadSettings(None)
        Settings_File = self.SimVars.Dir + "Settings.txt"
        Global_Settings_File = os.path.expanduser("~") + "/SpyDenSettings.txt"
        if os.path.exists(Settings_File): os.remove(Settings_File)
        if os.path.exists(Global_Settings_File): os.remove(Global_Settings_File)

    def clear_stuff(self,RePlot):
        """Clear and reset various components and data.

        The method clears and resets various components and data used in the application. It deactivates
        specific buttons, hides specific UI elements, clears the lists `SpineArr` and `DendArr`, removes
        ROI interactors, disconnects the spine marker, clears the plot, and resets the state of the
        delete old result button.

        Returns:
            None
        """


        self.add_commands([])
        self.PunctaCalc = False
        for button in [self.dendritic_width_button,
                        self.spine_button,self.spine_button_NN,
                        self.spine_button_ROI,self.delete_old_result_button,self.measure_spine_button,
                        self.spine_bg_button,self.old_ROI_button,self.measure_puncta_button,self.save_button]:
            MakeButtonInActive(button)
        self.hide_stuff([self.Dend_shift_check])
            
        self.SpineArr = []
        self.DendArr  = []
        self.punctas  = []
        self.DendMeasure  = []
        try:
            del self.DendMeasure
        except:
            pass
        if(hasattr(self,"roi_interactor_list")):
            for R in self.roi_interactor_list:
                R.clear()
        if(hasattr(self,"roi_interactor_list_bg")):
            for R in self.roi_interactor_list_bg:
                R.clear()
        if(hasattr(self,"line_interactor_list")):
            for L in self.line_interactor_list:
                L.clear()
            del self.line_interactor_list
        if(hasattr(self,"line_interactor_list")):
            for l in self.line_interactor_list:
                l.remove()
            del self.line_interactor_list
        self.roi_interactor_list = []
        self.roi_interactor_list_bg = []
        try:
            self.spine_marker.disconnect
            del self.spine_marker
        except:
            pass
        self.mpl.clear_plot()
        if(RePlot):
            try:
                self.update_plot_handle(
                    self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
                )
            except:
                pass
        else:
            self.tiff_Arr = np.zeros(shape=(1024, 1024))
            self.update_plot_handle(
                    self.tiff_Arr
                )
        self.delete_old_result_button.setChecked(False)

    def clear_old(self):

        """Clear and reset various components and data.

        The method clears and resets various components and data used in the application. It deactivates
        specific buttons, hides specific UI elements, clears the lists `SpineArr` and `DendArr`, removes
        ROI interactors, disconnects the spine marker, clears the plot, and resets the state of the
        delete old result button.

        Returns:
            None
        """

        self.clear_stuff(True)
        self.clear_settings()

        Dend_Dir = self.SimVars.Dir + "Dendrite/"
        if os.path.exists(Dend_Dir):
            shutil.copytree(Dend_Dir, Dend_Dir[:-1]+'temp/')
            shutil.rmtree(Dend_Dir)
        Spine_Dir = self.SimVars.Dir + "Spine/"
        if os.path.exists(Spine_Dir):
            shutil.copytree(Spine_Dir, Spine_Dir[:-1]+'temp/')
            shutil.rmtree(Spine_Dir)

    
    def on_projection_changed(self):

        """Handles the change of projection selection.

        The method is triggered when the user changes the selection of the projection option.
        It retrieves the new projection value, handles the editing finish for channel 0, and
        handles the editing finish for channel 1, if required.

        Returns:
            None
        """

        self.SimVars.GetNumpyFunc(self.projection.currentText())
        multwin = self.multiwindow_check.isChecked()
        res = self.res.text()
        if(res==""):
            res = 0

        self.tiff_Arr_Raw, self.SimVars.Times, meta_data, scale = GetTiffData(None, float(res), self.SimVars.z_type, self.SimVars.Dir,
                                                            Channels=multwin)
        self.tiff_Arr = np.copy(self.tiff_Arr_Raw)

        if (self.SimVars.Snapshots>1):
            self.tiff_Arr = GetTiffShift(self.tiff_Arr_Raw, self.SimVars)
            self.tiff_Arr_glob = np.copy(self.tiff_Arr)
            self.SimVars.xLims = self.SimVars.xLimsG
            self.SimVars.yLims = self.SimVars.yLimsG
        else:
            self.tiff_Arr = self.tiff_Arr_Raw
        self.mpl.update_plot(self.tiff_Arr[self.actual_timestep, self.actual_channel])

    def on_analyze_changed(self):

        """Handles the change of analysis option.

        The method is triggered when the user changes the selection of the analysis option.
        It updates the analysis mode in the SimVars object, deactivates the measure spine
        button and spine background button, and handles the editing finish for channel 1.

        Returns:
            None
        """

        self.SimVars.Mode = self.analyze.currentText()
        MakeButtonInActive(self.measure_spine_button)
        MakeButtonInActive(self.spine_bg_button)
        self.handle_editing_finished(1)
        
    def LoadSettings(self,Settings_File):
        if(Settings_File is not None):
            try:
                with open(Settings_File, "r") as file:
                    # Read the lines of the file
                    lines = file.readlines()

                # Process the lines
                for line in lines:
                    # Split each line into key-value pairs
                    if("MLLocation" in line):
                        value = line[11:-1]
                        if(os.path.isfile(value)):
                            self.NN_path = value
                            self.NN = True
                            self.button_set_NN.setText("Set NN! (saved)")
                        elif(os.path.isfile(self.NN_path)):
                            self.NN_path = self.NN_path
                            self.NN = True
                            self.button_set_NN.setText("Set NN! (default)")
                        else:
                            self.NN = False
                            self.button_set_NN.setText("Download NN")
                            self.button_set_NN.disconnect()
                            self.button_set_NN.clicked.connect(self.download_NN)
                    else:
                        key, value = line.strip().split(":")
                        if(key=="multi-time"):
                            boolean_value = value == "True"
                            self.multitime_check.setChecked(boolean_value)
                            self.SimVars.multitime_flag = boolean_value
                            if(self.SimVars.multitime_flag):
                                self.timestep_slider.setVisible(True)
                                self.timestep_counter.setVisible(True)
                                self.timestep_label.setVisible(True)
                        elif(key=="Analysis mode"):
                            self.analyze.setCurrentText(value)
                            self.SimVars.Mode = value
                        elif(key=="Dend. shift"):
                            boolean_value = value == "True"
                            self.Dend_shift_check.setChecked(boolean_value)
                        elif(key=="resolution"):
                            scale = float(value)
                        elif(key == "Puncta sigma range"):
                            sigma_r = tuple(map(int, value.split('-')))
                            self.puncta_sigma_range_slider.setValue(sigma_r)
                        else:
                            value = int(value)
                            if(key=="Image threshold"):
                                self.thresh_slider.setValue(value)
                            elif(key=="Dendritic width"):
                                self.neighbour_slider.setValue(value)
                                self.neighbour_counter.setText(str(value))
                            elif(key=="ML Confidence"):
                                self.ml_confidence_slider.setValue(value)
                                self.confidence_counter.setText(str(value))
                            elif(key=="ROI Tolerance"):
                                self.tol_val = value
                                self.tolerance_slider.setValue(value)
                                self.tolerance_counter.setText(str(value))
                            elif(key=="ROI Sigma"):
                                self.sigma_val = value
                                self.sigma_slider.setValue(value)
                                self.sigma_counter.setText(str(value))
                            elif(key=="Dendritic puncta threshold"):
                                self.puncta_dend_slider.setValue(value)
                                self.puncta_dend_counter.setText(str(value))
                            elif(key=="Somatic puncta threshold"):
                                self.puncta_soma_slider.setValue(value)
                                self.puncta_soma_counter.setText(str(value))
                            elif(key=="Dend. width multiplier"):
                                self.dend_width_mult_slider.setValue(value)
                                dend_factor = "{:.1f}".format(self.get_actual_multiple_factor())
                                self.dend_width_mult_counter.setText(dend_factor)
                            elif(key=="Neck Sigma"):
                                self.spine_neck_sigma_slider.setValue(value)
                                self.spine_neck_sigma_counter.setText(str(value))
                            elif(key=="Neck width multiplier"):
                                self.spine_neck_width_mult_slider.setValue(value)
                                neck_factor = "{:.1f}".format(0.1*value)
                                self.spine_neck_width_mult_counter.setText(value)
            except Exception as e:
                self.set_status_message.setText('There was a problem with the settings file')
                if DevMode: print(e)
        else:
            try:
                self.neighbour_slider.disconnect()
                self.thresh_slider.disconnect()
                self.ml_confidence_slider.disconnect()
                self.sigma_slider.disconnect()
                self.tolerance_slider.disconnect()
                self.puncta_dend_slider.disconnect()
                self.puncta_soma_slider.disconnect()
                self.dend_width_mult_slider.disconnect()
                self.spine_neck_sigma_slider.disconnect()
                self.spine_neck_width_mult_slider.disconnect()
                self.Dend_shift_check.disconnect()
                self.puncta_sigma_range_slider.disconnect()
                self.analyze.disconnect()
            except Exception as e:
                pass

            self.analyze.setCurrentText("Luminosity")
            self.multitime_check.setChecked(False)
            self.SimVars.multitime_flag = False


            self.Dend_shift_check.setChecked(False)

            self.puncta_sigma_range_slider.setValue((1,2))

            self.default_thresh = int(np.mean(self.tiff_Arr[0, 0, :, :]))

            self.thresh_slider.setValue(self.default_thresh)


            self.neighbour_slider.setValue(6)
            self.neighbour_counter.setText(str(6))

            self.ml_confidence_slider.setValue(5)
            self.confidence_counter.setText(str(5))

            self.tol_val = 5
            self.tolerance_slider.setValue(self.tol_val)
            self.tolerance_counter.setText(str(self.tol_val))

            self.sigma_val = 5
            self.sigma_slider.setValue(self.sigma_val)
            self.sigma_counter.setText(str(self.sigma_val))

            self.puncta_dend_slider.setValue(12)
            self.puncta_dend_counter.setText(str(12))

            self.puncta_soma_slider.setValue(12)
            self.puncta_soma_counter.setText(str(12))

            self.dend_width_mult_slider.setValue(5)
            dend_factor = "{:.1f}".format(self.get_actual_multiple_factor())
            self.dend_width_mult_counter.setText(dend_factor)

            self.spine_neck_sigma_slider.setValue(6)
            self.spine_neck_sigma_counter.setText(str(6))

            self.spine_neck_width_mult_slider.setValue(5)
            neck_factor = "{:.1f}".format(0.5)
            self.spine_neck_width_mult_counter.setText(neck_factor)

            self.neighbour_slider.valueChanged.connect(self.dendritic_width_eval)
            self.thresh_slider.valueChanged.connect(self.dend_thresh)
            self.ml_confidence_slider.valueChanged.connect(self.thresh_NN)
            self.sigma_slider.valueChanged.connect(self.spine_tolerance_sigma)
            self.tolerance_slider.valueChanged.connect(self.spine_tolerance_sigma)
            self.puncta_soma_slider.valueChanged.connect(self.soma_threshold_slider_update)
            self.puncta_dend_slider.valueChanged.connect(self.dend_threshold_slider_update)
            self.puncta_sigma_range_slider.valueChanged.connect(self.puncta_sigma_slider_update)
            self.Dend_shift_check.stateChanged.connect(lambda state: self.check_changed(state,3))
            self.dend_width_mult_slider.valueChanged.connect((self.dendritic_width_changer))
            self.spine_neck_sigma_slider.valueChanged.connect((self.spine_measure))
            self.spine_neck_width_mult_slider.valueChanged.connect((self.spine_measure))
            self.analyze.currentTextChanged.connect(self.on_analyze_changed)
            try:
                self.SimVars.Unit = float(self.res.text())
            except: 
                self.SimVars.Unit = None
            return None

        return scale

    def handle_editing_finished(self,indx,CallTwice=False):
        """Handles the behaviour of the GUI after the folder and resolution are chosen.

        Args:
            indx (int): the flag to choose between the two input fields
            CallTwice (bool, optional): Indicates if the method is called twice. Defaults to False.

        Returns:
            None
        """

        Dir = self.folderpath
        cell = self.cell.currentText()
        Mode = self.analyze.currentText()
        self.multiwindow_check.setChecked(True)
        multwin = self.multiwindow_check.isChecked()

        res = self.res.text()
        if(res==""):
            res = 0
        projection = self.projection.currentText()
        instance = self

        if(indx==0):
            self.SpinesMeasured = False
            try:
                self.neighbour_slider.disconnect()
                self.thresh_slider.disconnect()
                self.ml_confidence_slider.disconnect()
                self.sigma_slider.disconnect()
                self.tolerance_slider.disconnect()
                self.puncta_dend_slider.disconnect()
                self.puncta_soma_slider.disconnect()
                self.dend_width_mult_slider.disconnect()
                self.spine_neck_width_mult_slider.disconnect()
                self.spine_neck_sigma_slider.disconnect()
                self.Dend_shift_check.disconnect()
                self.puncta_sigma_range_slider.disconnect()
                self.analyze.disconnect()
            except Exception as e:
                pass
            self.SimVars = Simulation(res, 0, Dir + "/" + cell + "/", 1, Mode, projection, instance)
            self.Dend_shift_check.setChecked(False)
            self.SimVars.multitime_flag = self.multitime_check.isChecked()
            self.SimVars.multiwindow_flag = self.multiwindow_check.isChecked()
            try:
                self.tiff_Arr_Raw, self.SimVars.Times, meta_data, scale = GetTiffData(None, float(res), self.SimVars.z_type, self.SimVars.Dir,
                                                                    Channels=multwin)
                self.tiff_Arr = np.copy(self.tiff_Arr_Raw)
                self.clear_stuff(True)
            except:
                self.clear_stuff(False)
                raise
            self.number_channels = self.tiff_Arr.shape[1]
            self.channel_slider.setMaximum(self.number_channels-1)
            self.channel_slider.setMinimum(0)
            self.channel_slider.setValue(0)
            self.channel_slider.setVisible(True)
            self.channel_counter.setVisible(True)
            self.channel_label.setVisible(True)

            self.number_timesteps = self.tiff_Arr.shape[0]
            self.timestep_slider.setMinimum(0)
            self.timestep_slider.setMaximum(self.number_timesteps - 1)
            self.timestep_slider.setValue(0)

            self.default_thresh = int(np.mean(self.tiff_Arr[0, 0, :, :]))
            self.thresh_slider.setMaximum(int(np.max(self.tiff_Arr[0, 0, :, :])))
            self.thresh_slider.setMinimum(int(np.mean(self.tiff_Arr[0, 0, :, :])))
            step = (np.max(self.tiff_Arr[0, 0, :, :])-np.mean(self.tiff_Arr[0, 0, :, :]))//100
            self.thresh_slider.setSingleStep(int(step))
            self.thresh_slider.setValue(self.default_thresh)

            self.update_plot_handle(self.tiff_Arr[0, 0])
            # Set parameters
            self.SimVars.Snapshots = meta_data[0]
            self.SimVars.Channels = meta_data[2]
            self.SimVars.bgmean = np.zeros([self.SimVars.Snapshots, self.SimVars.Channels])

            Settings_File = self.SimVars.Dir + "Settings.txt"
            Global_Settings_File = os.path.expanduser("~") + "/SpyDenSettings.txt"
            if(os.path.exists(Settings_File)):
                scale = self.LoadSettings(Settings_File)
            elif(os.path.exists(Global_Settings_File)):
                scale = self.LoadSettings(Global_Settings_File)

            self.SimVars.model = self.NN_path
            self.neighbour_slider.valueChanged.connect(self.dendritic_width_eval)
            self.thresh_slider.valueChanged.connect(self.dend_thresh)
            self.ml_confidence_slider.valueChanged.connect(self.thresh_NN)
            self.sigma_slider.valueChanged.connect(self.spine_tolerance_sigma)
            self.tolerance_slider.valueChanged.connect(self.spine_tolerance_sigma)
            self.puncta_soma_slider.valueChanged.connect(self.soma_threshold_slider_update)
            self.puncta_dend_slider.valueChanged.connect(self.dend_threshold_slider_update)
            self.puncta_sigma_range_slider.valueChanged.connect(self.puncta_sigma_slider_update)
            self.multitime_check.stateChanged.connect(lambda state: self.check_changed(state,1))
            self.multiwindow_check.stateChanged.connect(lambda state: self.check_changed(state,0))
            self.Dend_shift_check.stateChanged.connect(lambda state: self.check_changed(state,3))
            self.dend_width_mult_slider.valueChanged.connect((self.dendritic_width_changer))
            self.spine_neck_sigma_slider.valueChanged.connect((self.spine_measure))
            self.spine_neck_width_mult_slider.valueChanged.connect((self.spine_measure))
            self.analyze.currentTextChanged.connect(self.on_analyze_changed)
            self.SimVars.Unit = scale
            # Get shifting of snapshots
            if (self.SimVars.Snapshots>1):
                self.tiff_Arr = GetTiffShift(self.tiff_Arr_Raw, self.SimVars)
                self.tiff_Arr_glob = np.copy(self.tiff_Arr)
                self.SimVars.xLims = self.SimVars.xLimsG
                self.SimVars.yLims = self.SimVars.yLimsG
            else:
                self.tiff_Arr = self.tiff_Arr_Raw

                # Get Background values
            for i in range(self.SimVars.Channels):
                self.SimVars.bgmean[:, i] = Measure_BG(self.tiff_Arr[:, i, :, :], self.SimVars.Snapshots, self.SimVars.z_type)

            self.mpl.image = self.tiff_Arr[0,0]

            self.multiwindow_check.setEnabled(True)
            self.multitime_check.setEnabled(True)
            self.projection.setEnabled(True)
            self.analyze.setEnabled(True)
            if(scale>0):
                self.res.setFocus()
                self.res.setText("%.3f" % scale)
        if(indx==1):
            if(not CallTwice):
                self.clear_stuff(True)
            try:
                self.SimVars.Unit = float(self.res.text())
            except:
                pass
            MakeButtonActive(self.medial_axis_path_button)
            self.CheckOldDend()
        self.mpl.canvas.setFocus()
        
    def get_actual_multiple_factor(self):
        return 0.1*self.dend_width_mult_slider.value()

    def CheckOldDend(self):
        """Checks for the existence of old dendrite data and updates the corresponding buttons and plots.

        The method checks for the existence of old dendrite data files and updates the dendrite objects,
        adds the dendrites to the plot, and activates the dendritic width, spine, and delete old result buttons.
        It also checks for the existence of old spine data files, reads the data if available, creates a spine marker,
        and activates the spine ROI button if applicable. Finally, it activates the old ROI button if the corresponding
        spine data files are found.

        Returns:
            None
        """
        MakeButtonInActive(self.old_ROI_button)
        DList = glob.glob(self.SimVars.Dir + "/Dendrite/Dendrite*.npy")
        if DList:
            self.DendArr = []
            for D in DList:
                Dend = Dendrite(self.tiff_Arr,self.SimVars)
                Dend.control_points = np.load(D) 
                self.DendArr.append(Dend)
            if (self.SimVars.Snapshots > 1):
                if(self.Dend_shift_check.isChecked()):
                    dMax = np.max([np.max(D.control_points,axis=0) for D in self.DendArr],axis=0)
                    dMin = np.min([np.min(D.control_points,axis=0) for D in self.DendArr],axis=0)
                    dX = np.clip([dMin[0]-20,dMax[0]+20],0,self.tiff_Arr_Raw.shape[-1])
                    dY = np.clip([dMin[1]-20,dMax[1]+20],0,self.tiff_Arr_Raw.shape[-2])
                    self.tiff_Arr_Dend = GetTiffShiftDend(self.tiff_Arr_Raw, self.SimVars,dX,dY)
                    self.tiff_Arr = np.copy(self.tiff_Arr_Dend)
                    self.SimVars.xLims = self.SimVars.xLimsD
                    self.SimVars.yLims = self.SimVars.yLimsD
                if(self.SimVars.multitime_flag):
                    self.show_stuff([self.Dend_shift_check])

            for Dend in self.DendArr:
                Dend.UpdateParams(self.tiff_Arr)
                if(len(self.SimVars.xLims)>0):
                    Dend.control_points = Dend.control_points+np.array([self.SimVars.yLims[0],self.SimVars.xLims[0]])
                Dend.complete_medial_axis_path = GetAllpointsonPath(Dend.control_points)[:, :]
                Dend.medial_axis = Dend.control_points
                Dend.thresh      = int(self.thresh_slider.value())
                pol = Polygon(
                Dend.control_points, fill=False, closed=False, animated=False
                )
                Dend.curvature_sampled = Dend.control_points
                Dend.length            = GetLength(Dend.complete_medial_axis_path)*self.SimVars.Unit
                self.mpl.axes.add_patch(pol)

            MakeButtonActive(self.dendritic_width_button)

            MakeButtonActive(self.spine_button)

            if self.NN: MakeButtonActive(self.spine_button_NN)

            MakeButtonActive(self.delete_old_result_button)
            SpineDir = self.SimVars.Dir+'Spine/'
            if os.path.isfile(SpineDir+'Synapse_l.json') or os.path.isfile(SpineDir+'Synapse_a.json'):
                self.SpineArr = ReadSynDict(SpineDir, self.SimVars)
                self.spine_marker = spine_eval(self.SimVars,np.array([S.location for S in self.SpineArr]),np.array([1 for S in self.SpineArr]),np.array([S.type for S in self.SpineArr]),False)
                self.spine_marker.disconnect()
                MakeButtonActive(self.spine_button_ROI)
                if(self.SpineArr[0].shift is None):
                    self.local_shift = False
                    self.local_shift_check.blockSignals(True)
                    self.local_shift_check.setChecked(False)
                    self.local_shift_check.blockSignals(False)

            if((os.path.isfile(SpineDir+'Synapse_l.json') and self.SimVars.Mode=="Luminosity") or
                (os.path.isfile(SpineDir+'Synapse_a.json') and self.SimVars.Mode=="Area" and self.SimVars.multitime_flag)
                or (os.path.isfile(SpineDir+'Synapse_l.json') and self.SimVars.Mode=="Area" and not self.SimVars.multitime_flag)):
                MakeButtonActive(self.old_ROI_button)
            self.set_status_message.setText(self.status_msg["10"])
            self.mpl.update_plot(self.tiff_Arr[self.actual_timestep, self.actual_channel])


    def spine_NN(self):
        """Performs spine detection using neural network.

        The method first updates the control points and medial axis for the dendrites if available.
        It then adds commands and shows/hides relevant GUI elements.
        The neural network is run to detect spines, and the resulting points, scores, and flags are stored in the SimVars object.
        If a spine marker already exists, it merges the existing and new spine points if they are close enough.
        Finally, it creates a new spine marker based on the updated spine points.

        Returns:
            None
        """
        self.SaveROIstoSpine()
        MakeButtonInActive(self.measure_puncta_button)
        self.PunctaCalc = False
        if(hasattr(self,'DendMeasure')):
            self.DendArr = self.DendMeasure.DendArr
            for Dend in self.DendArr:
                Dend.control_points = Dend.lineinteract.getPolyXYs().astype(int)
                Dend.complete_medial_axis_path = GetAllpointsonPath(Dend.control_points)[:, :]
                Dend.medial_axis = Dend.control_points
                Dend.curvature_sampled = Dend.control_points
        self.add_commands(["Spine_Desc","Spine_Func","NN_Conf"])
        self.show_stuff_coll(["NN"])

        val = self.ml_confidence_slider.value() / 10
        self.confidence_counter.setText(str(val))

        points, scores = RunNN(self.SimVars,np.vstack([Dend.control_points for Dend in self.DendArr]), self.tiff_Arr[self.actual_timestep,self.actual_channel])

        self.SimVars.points_NN = points
        self.SimVars.scores_NN = scores
        self.SimVars.flags_NN = np.zeros_like(scores)
        if(hasattr(self,'spine_marker')):
            if(len(self.spine_marker.points)>0):
                delete_list = []
                old_points = self.spine_marker.points
                old_scores = self.spine_marker.scores
                old_flags  = self.spine_marker.flags
                for i,pt in enumerate(points):
                    if((np.linalg.norm(pt-old_points,axis=-1)<5).any()):
                        delete_list.append(i)
                new_p = [points[i] for i in range(len(points)) if i not in delete_list]
                new_s = [scores[i] for i in range(len(scores)) if i not in delete_list]
                new_f = [0]*len(new_s)
                new_points = np.array(new_p + old_points.tolist()).astype(int)
                new_scores = np.array(new_s + old_scores.tolist())
                new_flags = np.array(new_f + old_flags.tolist())
                self.SimVars.points_NN = new_points
                self.SimVars.scores_NN = new_scores
                self.SimVars.flags_NN  = new_flags
        self.spine_marker = spine_eval(SimVars=self.SimVars, points=self.SimVars.points_NN[self.SimVars.scores_NN>val],
            scores=self.SimVars.scores_NN[self.SimVars.scores_NN>val],flags=self.SimVars.flags_NN[self.SimVars.scores_NN>val])

        SpineDir = self.SimVars.Dir+'Spine/'
        MakeButtonActive(self.spine_button_ROI)
        if((os.path.isfile(SpineDir+'Synapse_l.json') and self.SimVars.Mode=="Luminosity") or
            (os.path.isfile(SpineDir+'Synapse_a.json') and self.SimVars.Mode=="Area" and self.SimVars.multitime_flag)):
            MakeButtonActive(self.old_ROI_button)

        self.spine_button_NN.setChecked(False)
        MakeButtonInActive(self.measure_spine_button)
        MakeButtonInActive(self.spine_bg_button)
        
    def add_commands(self, l: list) -> None:
        self.command_box.clear()
        self.command_box.appendPlainText("Functionality:")
        for i in l:
            self.command_box.appendPlainText(self.command_list[i])

    def thresh_NN(self):
        """Applies a threshold to the neural network scores and updates the spine marker.

        The method retrieves the threshold value from the confidence slider and updates the corresponding GUI element.
        It then displays a spinemarker based on the spine points with scores above the threshold.

        Returns:
            None
        """
        val = self.ml_confidence_slider.value() / 10
        self.confidence_counter.setText(str(val))

        if(hasattr(self,'spine_marker')):
            ps = self.spine_marker.SimVars.points_NN
            ss = self.spine_marker.SimVars.scores_NN
            fs = self.spine_marker.SimVars.flags_NN

        self.spine_marker = spine_eval(SimVars=self.SimVars, points=ps[ss>val],scores=ss[ss>val],flags=fs[ss>val])
    
    def spine_eval_handle(self) -> None:
        """
        calculation of the spine locations
        Returns: None

        """

        self.SaveROIstoSpine()
        MakeButtonInActive(self.measure_puncta_button)
        self.PunctaCalc = False
        if(hasattr(self,'DendMeasure')):
            self.DendArr = self.DendMeasure.DendArr
            for Dend in self.DendArr:
                Dend.control_points = Dend.lineinteract.getPolyXYs().astype(int)
                Dend.complete_medial_axis_path = GetAllpointsonPath(Dend.control_points)[:, :]
                Dend.medial_axis = Dend.control_points
                Dend.curvature_sampled = Dend.control_points
        if(hasattr(self,'spine_marker')):
            old_points = self.spine_marker.points
            old_scores = self.spine_marker.scores
            old_flags = self.spine_marker.flags
        else:
            old_points = np.array([])
            old_scores = np.array([])
            old_flags = np.array([])
        self.show_stuff_coll([])
        self.add_commands(["Spine_Desc","Spine_Func"])
        self.spine_marker = spine_eval(SimVars=self.SimVars,points=old_points,scores=old_scores,flags=old_flags)
        self.spine_button.setChecked(False)
        MakeButtonInActive(self.measure_spine_button)
        MakeButtonInActive(self.spine_bg_button)
        
        MakeButtonActive(self.spine_button_ROI)
        SpineDir = self.SimVars.Dir+'Spine/'
        if((os.path.isfile(SpineDir+'Synapse_l.json') and self.SimVars.Mode=="Luminosity") or
            (os.path.isfile(SpineDir+'Synapse_a.json') and self.SimVars.Mode=="Area" and self.SimVars.multitime_flag)):
            MakeButtonActive(self.old_ROI_button)

    def dendritic_width_eval(self) -> None:
        """
        function that performs the dendritic width calculation
        when the button is pressed
        Returns: None

        """
        self.set_status_message.setText('Calculating Dendritic width')
        QCoreApplication.processEvents()
        self.SimVars.frame.set_status_message.repaint()
        self.PunctaCalc = False
        if(hasattr(self,'DendMeasure')):
            self.DendArr = self.DendMeasure.DendArr
            for Dend in self.DendArr:
                Dend.control_points = Dend.lineinteract.getPolyXYs().astype(int)
                Dend.complete_medial_axis_path = GetAllpointsonPath(Dend.control_points)[:, :]
                Dend.medial_axis = Dend.control_points
                Dend.curvature_sampled = Dend.control_points
        self.add_commands(["Width_Desc"])
        self.show_stuff_coll(["DendWidth"])
        dend_factor = self.get_actual_multiple_factor()
        dend_factor_str = "{:.2f}".format(dend_factor)
        self.dend_width_mult_counter.setText(dend_factor_str)
        self.neighbour_counter.setText(str(self.neighbour_slider.value()))
        if(hasattr(self.DendArr[0],'lineinteract')):
            for D in self.DendArr:
                D.lineinteract.clear()
        self.mpl.clear_plot()
        try:
            self.update_plot_handle(
                self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
            )
        except:
            pass
        for i,D in enumerate(self.DendArr): 
            self.set_status_message.setText('Calculating Dendritic width, Dendrite: '+str(i+1))
            QCoreApplication.processEvents()
            self.SimVars.frame.set_status_message.repaint()
            D.actual_channel  = self.actual_channel
            D.actual_timestep = self.actual_timestep
            D.WidthFactor     = dend_factor
            D.set_surface_contours(
                max_neighbours=5, sigma=self.neighbour_slider.value(), width_factor=dend_factor
            )
            dend_surface = D.get_dendritic_surface_matrix()
            dend_cont = D.get_contours()
            polygon = np.array(dend_cont[0][:, 0, :])
            pol = Polygon(dend_cont[0][:, 0, :], fill=False, closed=True,color='y')
            self.mpl.axes.add_patch(pol)
            self.mpl.canvas.draw()

        MakeButtonActive(self.save_button)
        MakeButtonActive(self.measure_puncta_button)
        self.dendritic_width_button.setChecked(False)

        self.set_status_message.setText('Width calculation complete')
        QCoreApplication.processEvents()
        self.SimVars.frame.set_status_message.repaint()

    def dendritic_width_changer(self) -> None:
        """
        function that multiplies the calculated dendrite segmentation by the correct
        Returns: None

        """
        dend_factor = self.get_actual_multiple_factor()
        dend_factor_str = "{:.2f}".format(dend_factor)

        self.dend_width_mult_counter.setText(dend_factor_str)
        self.neighbour_counter.setText(str(self.neighbour_slider.value()))

        if(hasattr(self.DendArr[0],'lineinteract')):
            for D in self.DendArr:
                D.lineinteract.clear()
        self.mpl.clear_plot()

        try:
            self.update_plot_handle(
                self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
            )
        except:
            pass
        for i,D in enumerate(self.DendArr): 

            D.actual_channel = self.actual_channel
            D.actual_timestep= self.actual_timestep
            D.OldWidthFactor  = D.WidthFactor
            D.WidthFactor     = dend_factor
            mask = np.zeros(shape=self.tiff_Arr[0,0].shape)

            for pdx, p in enumerate(D.smoothed_all_pts):
                mask = D.GenEllipse(mask,p,pdx,D.dend_stat[:, 4], D.dend_stat[:, 2]*dend_factor/D.OldWidthFactor,self.actual_timestep,self.actual_channel)

            gaussian_mask = (gaussian_filter(input=mask, sigma=self.neighbour_slider.value()) >= np.mean(mask)).astype(np.uint8)
            D.contours, _ = cv.findContours(gaussian_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            D.dendritic_surface_matrix= gaussian_mask
            polygon = np.array(D.contours[0][:, 0, :])
            pol = Polygon(D.contours[0][:, 0, :], fill=False, closed=True,color='y')
            self.mpl.axes.add_patch(pol)
            self.mpl.canvas.draw()
    
    def medial_axis_eval_handle(self) -> None:
        """
        performs the medial axis calcultation
        Returns: None
        """
        #save_window_pdf(self,'DataAnalWindow',2)
        self.PunctaCalc = False
        self.show_stuff_coll(["MedAx"])

        self.add_commands(["MP_Desc","MP_line"])
        if(hasattr(self,"DendMeasure")):
            self.DendArr = self.DendMeasure.DendArr
        if(hasattr(self,"spine_marker")):
            self.spine_marker.disconnect()
        try:
            self.mpl.clear_plot()
            self.default_thresh = self.thresh_slider.value()
            image = self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
            self.mpl.update_plot((image>=self.default_thresh)*image)
        except:
            pass

        self.DendMeasure= medial_axis_eval(self.SimVars,self.DendArr,self)
        self.medial_axis_path_button.setChecked(False)

    def get_path(self) -> None:
        """
        opens a dialog field where you can select the folder
        Returns: None

        """
        path = QFileDialog.getExistingDirectory(self, "Select Folder!")
        if(path):
            self.cell.clear()
            self.folderpath = path
            self.folderpath_label.setText(str(self.folderpath))
            self.set_status_message.setText(self.status_msg["1"])
            self.cell.setEnabled(True)
            self.res.setEnabled(True)
            choices = sorted(os.listdir(self.folderpath))
            for choice in choices:
                if(os.path.isdir(self.folderpath+'/'+choice)):
                    self.cell.addItem(choice)
        self.folderpath_button.setChecked(False)



    def dend_thresh(self):

        """Applies a threshold to the dendrite image and updates the plot.

        The method retrieves the threshold value from the threshold slider and updates the default threshold.
        It then applies the threshold to the current dendrite image and updates the plot.
        If a dendrite measurement object exists, it also updates the threshold value in the object.

        Returns:
            None
        """

        image = self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
        MakeButtonInActive(self.dendritic_width_button)
        self.default_thresh = self.thresh_slider.value()
        if(hasattr(self,"DendMeasure")):
            self.DendMeasure.thresh = self.default_thresh
            self.DendMeasure.DendClear(self.tiff_Arr)
        else:
            self.mpl.clear_plot()
            self.mpl.update_plot((image>=self.default_thresh)*image)

    def change_channel(self,value) -> None:
        """Handles the change of channel by updating relevant GUI elements and the plot.

        The method updates the maximum value of the channel slider, retrieves the selected channel,
        updates the channel counter, and adjusts the threshold slider based on the mean and maximum values of the channel.
        It then updates the plot with the new channel image.
        For puncta related stuff, it also removes the other channel punctas and add current channel punctas

        Returns:
            None
        """

        self.channel_slider.setMaximum(self.tiff_Arr.shape[1] - 1)
        self.actual_channel = self.channel_slider.value()
        self.channel_counter.setText(str(self.actual_channel))

        self.previous_timestep = self.actual_timestep
        self.timestep_slider.setMaximum(self.tiff_Arr.shape[0] - 1)
        self.actual_timestep = self.timestep_slider.value()
        self.timestep_counter.setText(str(self.actual_timestep))
        if(hasattr(self,"DendMeasure")):
            if(len(self.DendMeasure.DendArr)==0):
                mean = np.mean(self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :])
                max = np.max(self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :])
                self.thresh_slider.blockSignals(True)
                self.thresh_slider.setMinimum(int(mean))
                self.thresh_slider.setMaximum(int(max))
                self.thresh_slider.setValue(int(mean))
                self.thresh_slider.blockSignals(False)
        else:
            mean = np.mean(self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :])
            max = np.max(self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :])
            self.thresh_slider.blockSignals(True)
            self.thresh_slider.setMinimum(int(mean))
            self.thresh_slider.setMaximum(int(max))
            self.thresh_slider.setValue(int(mean))
            self.thresh_slider.blockSignals(False)

        try:
            if(hasattr(self,"ContourLines") and (self.SimVars.Mode == "Luminosity" and self.local_shift) or (self.SimVars.Mode == "Area" and self.SimVars.multitime_flag)):
                for i,l in enumerate(self.ContourLines):
                    if(self.SpineArr[i].type<2):
                        l.set_data(self.SpineArr[i].neck_contours[self.actual_timestep][:,0], self.SpineArr[i].neck_contours[self.actual_timestep][:,1])
        except:
            pass

        self.update_plot_handle(
            self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :])
        if(self.PunctaCalc):
            self.display_puncta()

        self.mpl.canvas.setFocus()
    
    def SaveROIstoSpine(self):

        for i,(R,L) in enumerate(zip(self.roi_interactor_list,self.line_interactor_list)):
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                try:
                    self.SpineArr[i].points = (R.poly.xy - R.shift[R.Snapshot]).tolist()[:-1]
                    self.SpineArr[i].shift[R.Snapshot] = R.shift[R.Snapshot]
                except:
                    self.SpineArr[i].points = (R.poly.xy).tolist()[:-1]
                if(self.SpineArr[i].type<2):
                    if(self.local_shift):
                        self.SpineArr[i].neck[R.Snapshot] = (L.poly.xy).tolist()
                    else:
                        self.SpineArr[i].neck = (L.poly.xy).tolist()
            else:
                if(self.local_shift):
                    self.SpineArr[i].points[self.actual_timestep] = (R.poly.xy - R.shift[R.Snapshot]).tolist()[:-1]
                else:
                    self.SpineArr[i].points[self.actual_timestep] = (R.poly.xy).tolist()[:-1]
                if(self.SpineArr[i].type<2):
                    
                    self.SpineArr[i].neck[R.Snapshot] = (L.poly.xy).tolist()
                try:
                    self.SpineArr[i].shift[R.Snapshot] = R.shift[R.Snapshot]
                except:
                    pass


    def update_plot_handle(self, image: np.ndarray) -> None:
        """
        updates the plot without destroying the Figure
        Args:
            image: np.array

        Returns:

        """
        try:
            if(self.SimVars.Mode=="Luminosity"):
                for i,(R,L) in enumerate(zip(self.roi_interactor_list,self.line_interactor_list)):
                    R.poly.xy = R.poly.xy - R.shift[R.Snapshot]
                    self.SpineArr[i].points = (R.poly.xy)[:-1].tolist()
                    OldSnapshot = R.Snapshot
                    R.Snapshot = self.actual_timestep

                    if(self.local_shift):
                        R.poly.xy = R.poly.xy + R.shift[R.Snapshot]
                        R.loc = [R.OgLoc[0]+R.shift[R.Snapshot][0],R.OgLoc[1]+R.shift[R.Snapshot][1]]
                        x_coord = R.OgLoc[0] + R.shift[R.Snapshot][0]
                        y_coord = R.OgLoc[1] + R.shift[R.Snapshot][1]
                        R.line_centre.set_data([x_coord], [y_coord])
                    else:                        
                        x_coord = R.OgLoc[0]
                        y_coord = R.OgLoc[1]
                        R.line_centre.set_data([x_coord], [y_coord])

                    R.line.set_data(zip(*R.poly.xy))
                    if(self.SpineArr[i].type<2):
                        if(self.local_shift):
                            self.SpineArr[i].shift[R.Snapshot] = R.shift[R.Snapshot]
                            self.SpineArr[i].neck[OldSnapshot] = (L.poly.xy).tolist()
                            L.poly.xy = self.SpineArr[i].neck[R.Snapshot]
                            L.line.set_data(zip(*L.poly.xy))
                        else:
                            L.poly.xy = L.poly.xy - R.shift[OldSnapshot]
                            self.SpineArr[i].neck = (L.poly.xy).tolist()
                            L.poly.xy = L.poly.xy + R.shift[OldSnapshot]
                            L.line.set_data(zip(*L.poly.xy))
                            L.poly.xy[0] = [x_coord,y_coord]

                    if(self.actual_timestep>0):
                        R.line_centre.set_color('r')
                        R.line_centre.set_markerfacecolor('k')
                    else:
                        R.line_centre.set_color('gray')
                        R.line_centre.set_markerfacecolor('gray')

            else:
                for i,(R,L) in enumerate(zip(self.roi_interactor_list,self.line_interactor_list)):
                    if(self.local_shift):
                        OldSnapshot = R.Snapshot
                        R.poly.xy = R.poly.xy - R.shift[R.Snapshot]
                        self.SpineArr[i].points[R.Snapshot] = (R.poly.xy)[:-1].tolist()
                        self.SpineArr[i].shift[R.Snapshot] = R.shift[R.Snapshot]
                        R.Snapshot = self.actual_timestep
                        newDat = np.array(self.SpineArr[i].points[R.Snapshot])+np.array(R.shift[R.Snapshot])
                        R.poly.xy = newDat
                        R.line.set_data([[newDat[:,0]],[newDat[:,1]]])
                        R.line_centre.set_data([[R.OgLoc[0]+R.shift[R.Snapshot][0]],[R.OgLoc[1]+R.shift[R.Snapshot][1]]])
                        R.loc = [R.OgLoc[0]+R.shift[R.Snapshot][0],R.OgLoc[1]+R.shift[R.Snapshot][1]]
                        R.points =  np.array(R.poly.xy)-np.array(R.loc)
                        if(self.SpineArr[i].type<2):
                            self.SpineArr[i].neck[OldSnapshot] = (L.poly.xy).tolist()
                            L.poly.xy = self.SpineArr[i].neck[R.Snapshot]
                            L.line.set_data(zip(*L.poly.xy))

                        if(self.actual_timestep>0):
                            R.line_centre.set_color('r')
                            R.line_centre.set_markerfacecolor('k')
                        else:
                            R.line_centre.set_color('gray')
                            R.line_centre.set_markerfacecolor('gray')
                    elif(not self.SimVars.multitime_flag):
                        OldSnapshot = R.Snapshot
                        self.SpineArr[i].points = (R.poly.xy)[:-1].tolist()
                        if(self.SpineArr[i].type<2):
                            self.SpineArr[i].neck[OldSnapshot] = (L.poly.xy).tolist()
                            L.poly.xy = self.SpineArr[i].neck[R.Snapshot]
                            L.line.set_data(zip(*L.poly.xy))
                    else: # Need to fix this as well TODO
                        OldSnapshot = R.Snapshot
                        self.SpineArr[i].points[R.Snapshot] = (R.poly.xy)[:-1].tolist()
                        R.Snapshot = self.actual_timestep
                        newDat = np.array(self.SpineArr[i].points[R.Snapshot])
                        R.poly.xy = newDat
                        R.line.set_data([[newDat[:,0]],[newDat[:,1]]])
                        if(self.SpineArr[i].type<2):
                            self.SpineArr[i].neck[OldSnapshot] = (L.poly.xy).tolist()
                            L.poly.xy = self.SpineArr[i].neck[R.Snapshot]
                            L.line.set_data(zip(*L.poly.xy))
        except Exception as e:
           # Print the error message associated with the exception
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            print(f"Error on line {tb[-1].lineno}: {e}")
        self.mpl.update_plot(image)
        
    
    def remove_plot(self) -> None:
        """
        destroys the whole matplotlib widget object
        Returns:

        """
        self.mpl.remove_plot()
        
    
    def show_stuff_coll(self,Names) -> None:
        """Hides the specified GUI elements.

        Args:
            stuff: A list of GUI elements to be hidden.

        Returns:
            None
        """

        self.hide_stuff([self.puncta_dend_label,self.puncta_dend_slider,self.puncta_dend_counter])
        self.hide_stuff([self.puncta_soma_label,self.puncta_soma_slider,self.puncta_soma_counter])
        self.hide_stuff([self.puncta_sigma_label, self.puncta_sigma_range_slider])
        self.hide_stuff([self.thresh_slider, self.thresh_label])
        self.hide_stuff([self.neighbour_counter,self.neighbour_slider,self.neighbour_label])
        self.hide_stuff([self.sigma_label,self.sigma_counter,self.sigma_slider,
            self.tolerance_label,self.tolerance_counter,self.tolerance_slider])
        self.hide_stuff([self.ml_confidence_label,self.ml_confidence_slider,self.confidence_counter ])
        self.hide_stuff([self.dend_width_mult_label, self.dend_width_mult_slider, self.dend_width_mult_counter])
        self.hide_stuff([self.local_shift_check])
        self.hide_stuff([self.spine_neck_width_mult_label, self.spine_neck_width_mult_slider, self.spine_neck_width_mult_counter])
        self.hide_stuff([self.spine_neck_sigma_label, self.spine_neck_sigma_slider, self.spine_neck_sigma_counter])

        for Name in Names:
            if(Name=="Puncta"):
                self.show_stuff([self.puncta_dend_label,self.puncta_dend_slider,self.puncta_dend_counter])
                self.show_stuff([self.puncta_soma_label,self.puncta_soma_slider,self.puncta_soma_counter])
                self.show_stuff([self.puncta_sigma_label, self.puncta_sigma_range_slider])

            if(Name=="MedAx"):
                self.show_stuff([self.thresh_slider, self.thresh_label])
            if(Name=="DendWidth"):
                self.show_stuff([self.neighbour_counter,self.neighbour_slider,self.neighbour_label,
                                 self.dend_width_mult_label, self.dend_width_mult_slider, self.dend_width_mult_counter])

            if(Name=="NN"):
                self.show_stuff([self.ml_confidence_label,self.ml_confidence_slider,self.confidence_counter ])
            if(Name=="SpineROI"):
                self.show_stuff([self.sigma_label,self.sigma_counter,self.sigma_slider,
                                self.tolerance_label,self.tolerance_counter,self.tolerance_slider])
                if(self.SimVars.multitime_flag):
                    self.show_stuff([self.local_shift_check])
            if(Name=="MeasureROI"):
                self.show_stuff([self.spine_neck_width_mult_label,self.spine_neck_width_mult_slider,self.spine_neck_width_mult_counter,
                                self.spine_neck_sigma_label,self.spine_neck_sigma_slider,self.spine_neck_sigma_counter])

    def hide_stuff(self,stuff) -> None:
        """Hides the specified GUI elements.

        Args:
            stuff: A list of GUI elements to be hidden.

        Returns:
            None
        """

        for s in stuff:
            s.hide()
    
    def show_stuff(self,stuff) -> None:
        """SHows the specified GUI elements.

        Args:
            stuff: A list of GUI elements to be shown.

        Returns:
            None
        """
        for s in stuff:
            s.show()
        
    
    def check_changed(self, state,flag):
        """Handles the change event of a checkbox.

        Args:
            state: The state of the checkbox. 2 indicates the checkbox is checked.
            flag: An integer flag to identify the checkbox.

        Returns:
            None
        """
        if state == 2: # The state is 2 when the checkbox is checked
            if(flag==0):
                self.SimVars.multiwindow_flag = True
                self.show_stuff([self.channel_label,self.channel_slider,self.channel_counter])
            elif(flag==1):
                self.SimVars.multitime_flag = True
                self.show_stuff([self.timestep_label,self.timestep_slider,self.timestep_counter])
                if(len(self.DendArr)>0):
                    self.show_stuff([self.Dend_shift_check])
                MakeButtonInActive(self.old_ROI_button)
                MakeButtonInActive(self.measure_spine_button)
                MakeButtonInActive(self.spine_bg_button)
            elif(flag==2):
                self.local_shift = True
                self.spine_ROI_eval()
            elif(flag==3):
                self.UpdateLims(0)
        else:
            if(flag==0):
                self.SimVars.multiwindow_flag = False
                self.hide_stuff([self.channel_label,self.channel_slider,self.channel_counter])
            elif(flag==1):
                self.SimVars.multitime_flag = False
                self.local_shift_check.blockSignals(True)
                self.local_shift_check.setChecked(False)
                self.local_shift = False
                self.local_shift_check.blockSignals(False)
                self.hide_stuff([self.timestep_label,self.timestep_slider,self.timestep_counter,self.local_shift_check,self.Dend_shift_check])
                MakeButtonInActive(self.old_ROI_button)
                MakeButtonInActive(self.measure_spine_button)
                MakeButtonInActive(self.spine_bg_button)
            elif(flag==2):
                self.local_shift = False
                self.spine_ROI_eval()
            elif(flag==3):
                self.UpdateLims(1)
                
    def UpdateLims(self,flag):
        self.mpl.clear_plot()
        if(flag==0):
            if(hasattr(self.SimVars,"xLimsD")):
                self.tiff_Arr = np.copy(self.tiff_Arr_Dend)
                self.SimVars.xLims = self.SimVars.xLimsD
                self.SimVars.yLims = self.SimVars.yLimsD
            elif(len(self.DendArr)>0):
                dMax = np.max([np.max(D.control_points,axis=0) for D in self.DendArr],axis=0)
                dMin = np.min([np.min(D.control_points,axis=0) for D in self.DendArr],axis=0)
                dX = np.clip([dMin[0]-20,dMax[0]+20],0,self.tiff_Arr_Raw.shape[-1])
                dY = np.clip([dMin[1]-20,dMax[1]+20],0,self.tiff_Arr_Raw.shape[-2])
                self.tiff_Arr_Dend = GetTiffShiftDend(self.tiff_Arr_Raw, self.SimVars,dX,dY)
                self.tiff_Arr = np.copy(self.tiff_Arr_Dend)
                self.SimVars.xLims = self.SimVars.xLimsD
                self.SimVars.yLims = self.SimVars.yLimsD
        elif(flag==1):
            if(hasattr(self.SimVars,"xLimsG")):
                self.tiff_Arr = np.copy(self.tiff_Arr_glob)
                self.SimVars.xLims = self.SimVars.xLimsG
                self.SimVars.yLims = self.SimVars.yLimsG
            else:
                self.tiff_Arr_glob = GetTiffShift(self.tiff_Arr, self.SimVars)
                self.tiff_Arr = np.copy(self.tiff_Arr_glob)
                self.SimVars.xLims = self.SimVars.xLimsG
                self.SimVars.yLims = self.SimVars.yLimsG
        image = self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
        try:
            self.update_plot_handle(
                self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
            )
        except:
            pass
        if(hasattr(self,'DendMeasure')):
            for Dend in self.DendMeasure.DendArr:
                    Dend.UpdateParams(self.tiff_Arr)
                    if(len(self.SimVars.xLims)>0):
                        if(flag==0):
                            Dend.control_points = Dend.control_points-np.array([self.SimVars.yLimsG[0],self.SimVars.xLimsG[0]])
                            Dend.control_points = Dend.control_points+np.array([self.SimVars.yLimsD[0],self.SimVars.xLimsD[0]])
                        elif(flag==1):
                            Dend.control_points = Dend.control_points-np.array([self.SimVars.yLimsD[0],self.SimVars.xLimsD[0]])
                            Dend.control_points = Dend.control_points+np.array([self.SimVars.yLimsG[0],self.SimVars.xLimsG[0]])
                    Dend.complete_medial_axis_path = GetAllpointsonPath(Dend.control_points)[:, :]
                    Dend.medial_axis = Dend.control_points
                    Dend.thresh      = int(self.thresh_slider.value())
                    Dend.pol = Polygon(
                    Dend.control_points, fill=False, closed=False, animated=False
                    )
                    Dend.curvature_sampled = Dend.control_points
                    Dend.length            = GetLength(Dend.complete_medial_axis_path)*self.SimVars.Unit
                    self.mpl.axes.add_patch(Dend.pol)
                    try:
                        Dend.lineinteract.poly = Dend.pol
                        Dend.lineinteract.line.set_data(zip(*Dend.pol.xy))
                    except:
                        pass
        elif(hasattr(self,'DendArr')):
            if(len(self.DendArr)>0):
                for Dend in self.DendArr:
                    Dend.UpdateParams(self.tiff_Arr)
                    if(len(self.SimVars.xLims)>0):
                        if(flag==0):
                            Dend.control_points = Dend.control_points-np.array([self.SimVars.yLimsG[0],self.SimVars.xLimsG[0]])
                            Dend.control_points = Dend.control_points+np.array([self.SimVars.yLimsD[0],self.SimVars.xLimsD[0]])
                        elif(flag==1):
                            Dend.control_points = Dend.control_points-np.array([self.SimVars.yLimsD[0],self.SimVars.xLimsD[0]])
                            Dend.control_points = Dend.control_points+np.array([self.SimVars.yLimsG[0],self.SimVars.xLimsG[0]])
                    Dend.complete_medial_axis_path = GetAllpointsonPath(Dend.control_points)[:, :]
                    Dend.medial_axis = Dend.control_points
                    Dend.thresh      = int(self.thresh_slider.value())
                    Dend.pol = Polygon(
                    Dend.control_points, fill=False, closed=False, animated=False
                    )
                    Dend.curvature_sampled = Dend.control_points
                    Dend.length            = GetLength(Dend.complete_medial_axis_path)*self.SimVars.Unit
                    self.mpl.axes.add_patch(Dend.pol)
                    try:
                        Dend.lineinteract.poly = Dend.pol
                        Dend.lineinteract.line.set_data(zip(*Dend.pol.xy))
                    except:
                        pass
        if(hasattr(self,'spine_marker')):
            if(len(self.spine_marker.points)>0):
                if(flag==0):
                    self.spine_marker.points = self.spine_marker.points-np.array([self.SimVars.yLimsG[0],self.SimVars.xLimsG[0]])
                    self.spine_marker.points = self.spine_marker.points+np.array([self.SimVars.yLimsD[0],self.SimVars.xLimsD[0]])
                elif(flag==1):
                    self.spine_marker.points = self.spine_marker.points-np.array([self.SimVars.yLimsD[0],self.SimVars.xLimsD[0]])
                    self.spine_marker.points = self.spine_marker.points+np.array([self.SimVars.yLimsG[0],self.SimVars.xLimsG[0]])


                self.spine_marker = spine_eval(self.SimVars, self.spine_marker.points,self.spine_marker.scores,self.spine_marker.flags,clear_plot=False)
                self.spine_marker.disconnect()

class DirStructWindow(QWidget):
    """Class that defines the directory structure window"""

    def __init__(self):
        super().__init__()

        self.title = "Folder generation Window"
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 300

        self.sourcepath = ""
        self.targetpath = ""
        self.FolderName = ""

        self.folderpath = "None"
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.grid = QGridLayout(self)
        # path button
        self.sourcepath_button = QPushButton(self)
        self.sourcepath_button.setText("Select source path!")
        self.sourcepath_button.clicked.connect(self.get_source)
        self.sourcepath_button.setToolTip('Provide the path to your expt. data')

        MakeButtonActive(self.sourcepath_button)
        self.grid.addWidget(self.sourcepath_button, 0, 0)
        self.sourcepath_label = QLineEdit(self)
        self.sourcepath_label.setReadOnly(True)
        self.sourcepath_label.setText(str(self.sourcepath))
        self.grid.addWidget(self.sourcepath_label, 0, 1)

        self.targetpath_button = QPushButton(self)
        self.targetpath_button.setText("Select target path!")
        self.targetpath_button.clicked.connect(self.get_target)
        MakeButtonInActive(self.targetpath_button)
        self.targetpath_button.setToolTip('Provide the path to where you want to copy it')
        self.grid.addWidget(self.targetpath_button, 1, 0)
        self.targetpath_label = QLineEdit(self)
        self.targetpath_label.setReadOnly(True)
        self.targetpath_label.setText(str(self.targetpath))
        self.grid.addWidget(self.targetpath_label, 1, 1)

        # name input
        self.FolderName = QLineEdit(self)
        self.grid.addWidget(self.FolderName, 2, 1)
        self.grid.addWidget(QLabel("Name of new folder (optional)"), 2, 0)
        self.FolderName.setEnabled(False)

        self.set_status_message = QLineEdit(self)
        self.set_status_message.setReadOnly(True)
        self.grid.addWidget(self.set_status_message, 3, 0, 1, 1)
        self.grid.addWidget
        self.set_status_message.setText("Select the data with your raw data")

        self.generate_button = QPushButton(self)
        self.generate_button.setText("Go!")
        MakeButtonInActive(self.generate_button)
        self.grid.addWidget(self.generate_button, 3, 1,1,1)
        self.generate_button.clicked.connect(self.generate_func)

    def get_source(self):
        """Allow user to select a directory and store it in global var called source_path"""

        self.sourcepath = QFileDialog.getExistingDirectory(self, "Select Folder!")
        if(self.sourcepath):
            self.sourcepath_label.setText(str(self.sourcepath))
            MakeButtonActive(self.targetpath_button)
            self.set_status_message.setText("Now select where you want to put the copy")
        self.sourcepath_button.setChecked(False)

    def get_target(self):
        """Allow user to select a directory and store it in global var called source_path"""

        self.targetpath = QFileDialog.getExistingDirectory(self, "Select Folder!")
        if(self.targetpath):
            self.targetpath_label.setText(str(self.targetpath))
            self.FolderName.setEnabled(True)
            MakeButtonActive(self.generate_button)
            self.set_status_message.setText("If you want a subfolder, give the name here")
        self.targetpath_button.setChecked(False)

    def generate_func(self):
        """Generate the target directory, and deleting the directory if it already exists"""
        # try:
        flag = GFS.CreateCellDirs(self.sourcepath, self.targetpath, self.FolderName.text())
        self.generate_button.setChecked(False)
        if flag == 0: 
            self.set_status_message.setText("Success: Your new folder exists")
        else:
            self.set_status_message.setText("There was a problem:"+str(flag))
    def get_path(self) -> None:
        """
        opens a dialog field where you can select the folder
        Returns: None

        """
        self.folderpath = QFileDialog.getExistingDirectory(self, "Select Folder!")
        self.folderpath_label.setText(str(self.folderpath))
        self.set_status_message.setText(self.status_msg["1"])

@handle_exceptions
class TutorialWindow(QWidget):
    """Class that defines the tutorial window"""

    def __init__(self):
        super().__init__()

        self.title = "Tutorial Window"
        self.left = 100
        self.top = 100
        self.width = 200
        self.height = 400

        self.foldurl = 'https://youtu.be/1o-l9o2W514'

        self.genurl = 'https://youtu.be/1DYjQp4MUGA'

        self.dendurl = 'https://youtu.be/bU41g8NW8Ts'

        self.spineurl = 'https://youtu.be/DiqYDdBQRz8'

        self.punctaurl = 'https://youtu.be/fgDD-Ucr3ms'

        self.filesurl = 'https://youtu.be/3QC2gGxzXi0'

        self.emailurl = 'mailto:meggl@umh.es?subject=SpyDen v'+version+' Bug // Feedback '

        self.Giturl = '<a href="https://github.com/meggl23/SpyDen">Github repo</a>'

        self.initUI()

    def initUI(self):


        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.grid = QGridLayout()
        self.setLayout(self.grid)  # Set the grid layout for DataReadWindow
        self.FolderTutorial_button = QPushButton(self)
        self.FolderTutorial_button.setText("How to generate folders")
        MakeButtonActive(self.FolderTutorial_button)
        self.FolderTutorial_button.clicked.connect((lambda: self.LoadURL(self.foldurl)))
        self.grid.addWidget(self.FolderTutorial_button, 0, 0, 1, 1)

        self.GenInfoTut_button = QPushButton(self)
        self.GenInfoTut_button.setText("General info")
        MakeButtonActive(self.GenInfoTut_button)
        self.GenInfoTut_button.clicked.connect((lambda: self.LoadURL(self.genurl)))
        self.grid.addWidget(self.GenInfoTut_button, 0, 1, 1, 1)

        self.DendriteTutorial_button = QPushButton(self)
        self.DendriteTutorial_button.setText("Analysing a dendrite")
        MakeButtonActive(self.DendriteTutorial_button)
        self.DendriteTutorial_button.clicked.connect((lambda: self.LoadURL(self.dendurl)))
        self.grid.addWidget(self.DendriteTutorial_button, 1, 0, 1, 1)

        self.SpineTutorial_button = QPushButton(self)
        self.SpineTutorial_button.setText("Analysing spines")
        MakeButtonActive(self.SpineTutorial_button)
        self.SpineTutorial_button.clicked.connect((lambda: self.LoadURL(self.spineurl)))
        self.grid.addWidget(self.SpineTutorial_button, 1, 1, 1, 1)

        self.PunctaTutorial_button = QPushButton(self)
        self.PunctaTutorial_button.setText("Analysing puncta")
        MakeButtonActive(self.PunctaTutorial_button)
        self.PunctaTutorial_button.clicked.connect((lambda: self.LoadURL(self.punctaurl)))
        self.grid.addWidget(self.PunctaTutorial_button, 2, 0, 1, 1)

        self.FileTutorial_button = QPushButton(self)
        self.FileTutorial_button.setText("The file structure")
        MakeButtonActive(self.FileTutorial_button)
        self.FileTutorial_button.clicked.connect((lambda: self.LoadURL(self.filesurl)))
        self.grid.addWidget(self.FileTutorial_button, 2, 1, 1, 1)

        
        h_layout2 = QHBoxLayout()


        label = QLabel(self.Giturl)
        label.setOpenExternalLinks(True) 
        label.setAlignment(Qt.AlignCenter)
        label.setMaximumHeight(30)
        label.linkActivated.connect(self.OpenLink)  # Connect to a slot to handle link activation
        label.setStyleSheet("QLabel { font-size: 20pt; }")

        self.grid.setRowStretch(3, 0)

        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        # Create and add the third button to the horizontal layout
        h_layout.addWidget(label)
        h_layout.addStretch(1)


        self.grid.addLayout(h_layout, 3, 0, 1, 2) 


        self.email_button = QPushButton(self)
        self.email_button.setText("Report a bug/provide feedback")
        self.email_button.clicked.connect((lambda: self.LoadURL(self.emailurl)))
        MakeButtonActive(self.email_button)
        # Create a horizontal layout for the second row
        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        # Create and add the third button to the horizontal layout
        h_layout.addWidget(self.email_button)
        h_layout.addStretch(1)


        self.grid.addLayout(h_layout, 4, 0, 1, 2) 

        self.setLayout(self.grid)

    def OpenLink(self, link_str):
        QDesktopServices.openUrl(QUrl(link_str))

    def LoadURL(self,url):

        wb.open(url)

        self.FolderTutorial_button.setChecked(False)
        self.DendriteTutorial_button.setChecked(False)
        self.SpineTutorial_button.setChecked(False)
        self.PunctaTutorial_button.setChecked(False)
        self.FileTutorial_button.setChecked(False)
        self.GenInfoTut_button.setChecked(False)
        self.email_button.setChecked(False)



class MainWindow(QWidget):
    """
    class that makes the main window
    """

    def __init__(self):
        super().__init__()
        self.title = "SpyDen"
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 600
        self.initUI()
        global DevMode

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color: rgb(0, 0, 0);")

        if(platform.system()=='Windows'):
            self.setWindowIcon(QIcon('App\\Brain.ico'))
        else:
            self.setWindowIcon(QIcon(QPixmap("brain.png")))
        self.grid = QGridLayout(self)

        # headline
        self.headline = QLabel(self)
        self.headline.setTextFormat(Qt.TextFormat.RichText)
        self.headline.setText("SpyDen <br> <font size='0.1'>v"+version+"-beta (Dev) </font>")
        Font = QFont("Courier", 60)
        self.headline.setFont(Font)
        self.headline.setStyleSheet("color: white")
        self.grid.addWidget(self.headline, 1, 1, 1, 6)

        # begin
        self.top_left = QLabel("          ", self)
        self.grid.addWidget(self.top_left, 0, 0, 1, 1)
        self.top_right = QLabel("          ", self)
        self.grid.addWidget(self.top_right, 0, 8, 1, 1)
        self.bottom_left = QLabel("          ", self)
        self.grid.addWidget(self.bottom_left, 10, 0, 1, 1)
        self.bottom_right = QLabel("          ", self)
        self.grid.addWidget(self.bottom_right, 10, 8, 1, 1)

        # image
        code_dir = os.path.dirname(os.path.abspath(__file__))

        # Relative path of the image file within the package structure
        relative_address = "dend.png"
        # Construct the absolute path of the image file within the package structure
        image_path_in_package = os.path.join(code_dir, relative_address)

        pixmap = QPixmap(image_path_in_package)
        pixmap = pixmap.scaled(1050, 410)
        self.image = QLabel(self)
        self.image.setPixmap(pixmap)
        self.grid.addWidget(self.image, 2, 1, 7, 6)

        # read data button
        self.read_data_button = QPushButton(self)
        self.read_data_button.setText("Read Data")
        MakeButtonActive(self.read_data_button,1)
        self.read_data_button.clicked.connect(self.read_data)
        self.grid.addWidget(self.read_data_button, 8, 1, 2, 2)
        self.read_data_button.setToolTip('Open the data analysis window')

        # Tutorial button
        self.tutorial_button = QPushButton(self)
        self.tutorial_button.setText("Tutorials")
        MakeButtonActive(self.tutorial_button,1)
        self.tutorial_button.clicked.connect(self.tutorial)
        self.grid.addWidget(self.tutorial_button, 8, 3, 2, 2)
        self.tutorial_button.setToolTip('Links to video tutorials')

        # Analyze button
        self.generate_button = QPushButton(self)
        self.generate_button.setText("Generate folders")
        MakeButtonActive(self.generate_button,1)
        self.generate_button.clicked.connect(self.generate)
        self.grid.addWidget(self.generate_button, 8, 5, 2, 2)
        self.generate_button.setToolTip('Format your files so SpyDen can read them')

        #========= multiwindow checkbox ================
        self.DevMode_check = QCheckBox(self)
        self.DevMode_check.setText("Developer Mode")
        self.grid.addWidget(self.DevMode_check, 7, 3, 1, 1)
        self.DevMode_check.stateChanged.connect(lambda state: self.check_changed(state))
        self.DevMode_check.setToolTip('Turn on Dev-mode, disables failsafes')


    def generate(self) -> None:

        self.gen_folder = DirStructWindow()
        self.gen_folder.show()
        self.generate_button.setChecked(False)

    def check_changed(self, state):
        """Handles the change event of a checkbox.

        Args:
            state: The state of the checkbox. 2 indicates the checkbox is checked.
            flag: An integer flag to identify the checkbox.

        Returns:
            None
        """
        global DevMode
        if state == 2: # The state is 2 when the checkbox is checked
            DevMode = True
        else:
            DevMode = False

    def tutorial(self) -> None:  # needs to be added

        """
        opens the Tutorial page
        Returns: None

        """

        self.tut_window = TutorialWindow()
        self.tut_window.show()
        self.tutorial_button.setChecked(False)

        pass

    def read_data(self) -> None:
        """
        opens the read data Window
        Args:
            checked:

        Returns:None

        """
        self.data_read = DataReadWindow()
        self.data_read.showMaximized()
        self.read_data_button.setChecked(False)

def save_window_pdf(Qw,name='temp',scale=10):

    """
    Function to save widget as PDF: 
    for intro:         save_window_pdf(self,'IntroWindow',10)
    for analysis:      save_window_pdf(self,'DataAnalWindow',5)
    """

    file_path = '/Users/maximilianeggl/Dropbox/PostDoc/ToolFigs/Figures/'+name+'.pdf'

    pixmap = QPixmap(1600, 1500)  # Use the size of the widget as the size of the pixmap
    pixmap.fill(Qt.white)  # Fill the pixmap with a white background
    painter = QPainter(pixmap)

    # Render the widget onto the QPixmap
    Qw.render(painter)
    painter.end()

    scaled_pixmap = pixmap.scaledToWidth(pixmap.width()*scale,Qt.SmoothTransformation)

    # Save the QPixmap to a PDF file using QPrinter
    printer = QPrinter(QPrinter.HighResolution)
    printer.setResolution(2400)
    printer.setOrientation(QPrinter.Landscape)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(file_path)

    pdf_painter = QPainter(printer)
    pdf_painter.drawPixmap(0, 0, scaled_pixmap)
    pdf_painter.end()

    sys.exit()

def RunWindow():
    app = QApplication(sys.argv)
    code_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative path of the image file within the package structure
    relative_address = "brain.png"
    # Construct the absolute path of the image file within the package structure
    image_path_in_package = os.path.join(code_dir, relative_address)
    app.setWindowIcon(QIcon(QPixmap(image_path_in_package)))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    RunWindow()
