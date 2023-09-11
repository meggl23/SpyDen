import os
import shutil
import glob

import numpy as np

from .MPL_Widget import *
from .DataRead import *
from matplotlib.widgets import Slider, Button

from . import GenFolderStruct as GFS

from .SynapseFuncs import FindShape
from .RoiInteractor import RoiInteractor,RoiInteractor_BG
from .PunctaDetection import save_puncta,PunctaDetection,Puncta
from .PathFinding import GetLength

import webbrowser as wb

DevMode = False

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

        code_dir = os.path.dirname(os.path.abspath(__file__))
        relative_address = "../MLModel/SynapseMLModel"
        self.NN_path = os.path.join(code_dir, relative_address)


        if(os.path.exists(self.NN_path)):
            self.NN = True
        else:
            self.NN = False

        self.SimVars = Simulation(0, 0, 0, 0,0,0)
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
            "11": "Calculating punctas"
        }

        self.command_list = {
        
        "MP_Desc":"Generate and edit the medial axis paths of dendrites. After every two clicks a path is generated \n"
        ,
        "MP_line": "Move to the start and end of the dendrite and press: \n"+
        "  - t                  - toggle between dragable vertices and lines \n" +
        "  - Left click    - mark the start and the dnd \n" + 
        "  - d                 - delete the first marker \n" +
        "  - backspace - clear the image \n" + 
        "The slider can be used to set the threshold of the image"
        ,
        "MP_vert":"Edit the dendrite path with draggable nodes:\n"+
        "  - t                  - toggle between dragable vertices and lines \n" +
        "  - d                 - delete a node\n" +
        "  - i                   - insert a Node \n"+
        "  - backspace - clear the image \n"
        "The slider can be used to set the threshold of the image"
        ,
        "Width_Desc":"Change the Dendritic Width Slider to adjust the Dendritic Width(s)",
        "Spine_Desc":"Generate spine locations either by clicking or using a neural network \n",
        "Spine_Func":"Move a Spine location and Press: \n"+
        "  - Left Click - to mark a Spine \n" +
        "  - d - to delete a marked Spine \n" + 
        "  - backspace - clear the image \n" + 
        "Hold shift (green) or control (yellow) and click to mark special spines (up to 3 different types of spines). To select a soma for the puncta detection, use control.",
        "NN_Conf":"Change the Confidence Slider to change the confidence of the NN",
        "SpineROI_Desc":"Edit the ROIs generated by the algorithm via draggable nodes\n",
        "SpineROI_Func":"Commands \n" +         
        "  - t                  - toggle between dragable vertices and lines \n" +
        "  - d                 - delete a node\n" +
        "  - i                   - insert a Node \n"+
        "  - backspace - clear the image \n"
        "The tolerance slider makes the ROIs bigger or smaller \n"
        "The sigma slider refers to the variance of the smoothing filter: smaller is for small sharp lines, larger for larger blurred lines",
        "SpineBG_Desc":"Edit the locations of the spine ROIs by dragging the red points for the local background calculation\n",
        "Puncta":"Use the sliders to calculate different puncta (surbhit - i will need your input on this). Puncta are automatically calculated on all channels and snapshots. It is possible that no puncta are found"
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

        #============= path input ==================
        self.cell = QLineEdit(self)
        self.cell.setEnabled(False)
        self.cell.editingFinished.connect(lambda: self.handle_editing_finished(0))
        self.grid.addWidget(self.cell, 1, 1, 1, 1)
        self.grid.addWidget(QLabel("Filename:"), 1, 0, 1, 1)

        #========= projection dropdown ===============
        self.projection = QComboBox(self)
        choices = ["Max","Min","Mean","Sum","Median","Std"]
        for choice in choices:
            self.projection.addItem(choice)
        self.grid.addWidget(self.projection, 3, 1, 1, 1)
        self.grid.addWidget(QLabel("Projection"), 3, 0, 1, 1)
        self.projection.setEnabled(False)
        self.projection.currentTextChanged.connect(self.on_projection_changed)

        #========= analyze dropdown ================
        self.analyze = QComboBox(self)
        choices = ["Luminosity", "Area"]
        for choice in choices:
            self.analyze.addItem(choice)
        self.grid.addWidget(self.analyze, 4, 1, 1, 1)
        self.grid.addWidget(QLabel("Analyze"), 4, 0, 1, 1)
        self.anal = self.projection.currentText()  # str variable what u want to analyze
        self.analyze.setEnabled(False)
        self.analyze.currentTextChanged.connect(self.on_analyze_changed)

        #========= multiwindow checkbox ================
        self.multiwindow_check = QCheckBox(self)
        self.multiwindow_check.setText("Multi Channel")
        self.multiwindow_check.setEnabled(False)
        self.SimVars.multiwindow_flag  = True
        self.grid.addWidget(self.multiwindow_check, 2, 0, 1, 1)

        #========= multiwindow checkbox ================
        self.multitime_check = QCheckBox(self)
        self.multitime_check.setText("Multi Time")
        self.multitime_check.setEnabled(False)
        self.SimVars.multitime_flag  = False
        self.grid.addWidget(self.multitime_check, 2, 1, 1, 1)

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
        self.spine_button.setText("Spine Localization Manually!")
        MakeButtonInActive(self.spine_button)
        self.grid.addWidget(self.spine_button, 10, 0, 1, 2)
        self.spine_button.clicked.connect(self.spine_eval_handle)

        #============= NN spine button ==================
        self.button_set_NN = QPushButton(self)
        if(self.NN):
            self.button_set_NN.setText("Set NN! (default)")
        else:
            self.button_set_NN.setText("Set NN! (default not found)")
        MakeButtonActive(self.button_set_NN)
        self.grid.addWidget(self.button_set_NN, 11, 0, 1, 1)
        self.button_set_NN.clicked.connect(self.set_NN)

        #============= NN spine button ==================
        self.spine_button_NN = QPushButton(self)
        self.spine_button_NN.setText("Spine Localization via NN!")
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
        self.measure_puncta_button.setText("Get and measure punctas")
        MakeButtonInActive(self.measure_puncta_button)
        self.grid.addWidget(self.measure_puncta_button, 15, 0, 1, 2)
        self.measure_puncta_button.clicked.connect(self.get_puncta)

        label = QLabel("Save/clear")
        label.setAlignment(Qt.AlignCenter)  # Centers the label horizontally
        label.setStyleSheet("font-size: 18px;")  # Increases the font size
        self.grid.addWidget(label, 16, 0, 1, 2)

        #============= delete button ==================
        self.delete_old_result_button = QPushButton(self)
        self.delete_old_result_button.setText("Clear all")
        self.grid.addWidget(self.delete_old_result_button, 17, 0, 1, 1)
        self.delete_old_result_button.clicked.connect(lambda: self.clear_stuff(True))
        MakeButtonInActive(self.delete_old_result_button)

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

        #============= dialog fields (commands) ==================
        self.command_box = QPlainTextEdit(self)
        self.command_box.setReadOnly(True)
        self.grid.addWidget(self.command_box, 19, 0, 1, 2)
        self.command_box.setFixedWidth(550)
        self.command_box.setFixedHeight(100)

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

        #============= threshold slider ==================
        self.thresh_label = QLabel("Threshold Value")
        self.grid.addWidget(self.thresh_label, 3, 8, 1, 1)
        self.thresh_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.thresh_slider, 3, 2, 1, 6)
        self.hide_stuff([self.thresh_slider,self.thresh_label])

        #============= dend width slider ==================
        self.neighbour_label = QLabel("Dendritic Width Smoothness")
        self.grid.addWidget(self.neighbour_label, 3, 8, 1, 1)
        self.neighbour_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.neighbour_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.neighbour_slider, 3, 2, 1, 6)
        self.neighbour_slider.setMinimum(0)
        self.neighbour_slider.setMaximum(10)
        self.neighbour_slider.setValue(6)
        self.neighbour_slider.singleStep()
        self.neighbour_counter = QLabel(str(self.neighbour_slider.value()))
        self.grid.addWidget(self.neighbour_counter, 3, 9, 1, 1)
        self.hide_stuff([self.neighbour_counter,self.neighbour_slider,self.neighbour_label])

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

        #============= spine sigma slider ==================
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

        #============= spine sigma slider ==================
        self.local_shift_check = QCheckBox(self)
        self.local_shift_check.setText("Local shifting")
        self.grid.addWidget(self.local_shift_check, 5, 2, 1, 1)
        self.local_shift_check.setVisible(False)
        self.local_shift = False
        self.local_shift_check.stateChanged.connect(lambda state: self.check_changed(state,2))
        
        # ============= Puncta dendritic threshold slider ==================
        self.puncta_dend_label = QLabel("Threshold dendrite")
        self.grid.addWidget(self.puncta_dend_label,3, 2, 1, 1)
        self.puncta_dend_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.puncta_dend_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.puncta_dend_slider,3 , 3, 1, 6)
        self.puncta_dend_slider.setMinimum(0)
        self.puncta_dend_slider.setMaximum(100)
        self.puncta_dend_slider.setValue(75)
        self.puncta_dend_slider.singleStep()

        self.puncta_dend_counter = QLabel(str(self.puncta_dend_slider.value()))
        self.grid.addWidget(self.puncta_dend_counter, 3, 9, 1, 1)


        # ============= Puncta soma threshold slider ==================
        self.puncta_soma_label = QLabel("Threshold soma")
        self.grid.addWidget(self.puncta_soma_label, 4, 2, 1, 1)
        self.puncta_soma_slider = ClickSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.puncta_soma_slider.setTickPosition(QSlider.TicksBelow)
        self.grid.addWidget(self.puncta_soma_slider, 4, 3, 1, 6)
        self.puncta_soma_slider.setMinimum(0)
        self.puncta_soma_slider.setMaximum(100)
        self.puncta_soma_slider.setValue(50)
        self.puncta_soma_slider.singleStep()

        self.puncta_soma_counter = QLabel(str(self.puncta_soma_slider.value()))
        self.grid.addWidget(self.puncta_soma_counter, 4, 9, 1, 1)

        self.hide_stuff([self.puncta_soma_label, self.puncta_soma_counter, self.puncta_soma_slider])
        self.hide_stuff([self.puncta_dend_label, self.puncta_dend_counter, self.puncta_dend_slider])

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
                xpert, shift, bgloc,closest_Dend = FindShape(
                tf,
                point,
                medial_axis_Arr,
                points,
                mean,
                True,
                sigma=self.sigma_val,
                tol=self.tol_val,
                SpineShift_flag = self.local_shift
                )
                polygon = np.array(xpert)
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                self.roi_interactor_list[index].poly = pol
                self.roi_interactor_list[index].line.set_data(pol.xy[:, 0], pol.xy[:, 1])
                self.SpineArr[index] = Synapse(list(point),list(bgloc),pts=xpert,shift=shift,
                    channel=self.actual_channel,Syntype=flag,closest_Dend=closest_Dend)
            else:
                self.SpineArr[index].points = []
                pt = self.SpineArr[index].location
                tiff_Arr_small = tf[:,
                                max(pt[1] - 50, 0) : min(pt[1] + 50, tf.shape[-2]),
                                max(pt[0] - 50, 0) : min(pt[0] + 50, tf.shape[-1]),
                            ]
                self.SpineArr[index].shift = SpineShift(tiff_Arr_small).T.astype(int).tolist()
                for i in range(self.SimVars.Snapshots):
                    xpert, _, bgloc,closest_Dend = FindShape(
                        tf[i],
                        np.array(self.SpineArr[index].location),
                        medial_axis_Arr,
                        points,
                        mean,
                        True,
                        sigma=self.sigma_val,
                        tol=self.tol_val,
                    )
                    self.SpineArr[index].points.append(xpert)
                    self.SpineArr[index].closest_Dend = closest_Dend
                polygon = np.array(self.SpineArr[index].points[self.actual_timestep])
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                self.roi_interactor_list[index].poly = pol
                self.roi_interactor_list[index].line.set_data(pol.xy[:, 0], pol.xy[:, 1])
            self.set_status_message.setText(self.set_status_message.text()+'.')
            QCoreApplication.processEvents()
            self.set_status_message.repaint()
            
        self.mpl.canvas.draw()

    def spine_measure(self):
        """
        function that takes the calculated ROIS and obtains various statistics
        Returns: None
        """
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
        for i,R in enumerate(self.roi_interactor_list):
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                self.SpineArr[i].points = (R.poly.xy - R.shift[R.Snapshot]).tolist()[:-1]
                try:
                    self.SpineArr[i].shift[R.Snapshot] = R.shift[R.Snapshot]
                except:
                    pass
            else:
                self.SpineArr[i].points[self.actual_timestep] = (R.poly.xy - R.shift[R.Snapshot]).tolist()[:-1]
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

        self.SpineArr = SynDistance(self.SpineArr, medial_axis_Arr, self.SimVars.Unit)

        Measure(self.SpineArr,self.tiff_Arr,self.SimVars,self)
        self.measure_spine_button.setChecked(False)
        MakeButtonActive(self.save_button)
        self.set_status_message.setText("Measuring ROI statistics")
        return None
    
    def dend_measure(self,Dend,i,Dend_Dir):
        """
        function that takes the calculated dendritic segments and saves these
        Returns: None
        """

        Dend_Struct_Dir = Dend_Dir + "Dend_Struct"+str(i)+".npy"
        Dend_Stat_Dir = Dend_Dir + "Dend_Path_Lumin"+str(i)+".npy"
        Dend_Save_Dir = Dend_Dir + "Dendrite"+str(i)+".npy"
        Dend_Mask_Dir = Dend_Dir + "/Mask_dend"+str(i)+".png"
        if(len(self.SimVars.yLims)>0):
            np.save(Dend_Save_Dir, Dend.control_points -
                np.array([self.SimVars.yLims[0],self.SimVars.xLims[0]]))
        else:
            np.save(Dend_Save_Dir, Dend.control_points)
        try:
            dend_mask = Dend.get_dendritic_surface_matrix() * 255
            cv.imwrite(Dend_Mask_Dir, dend_mask)
            if(len(self.SimVars.yLims)>0):
                Dend.dend_stat[:,:2]  = Dend.dend_stat[:,:2] + np.array([self.SimVars.yLims[0],self.SimVars.xLims[0]])
            np.save(Dend_Struct_Dir, Dend.dend_stat)
            if(len(self.SimVars.yLims)>0):
                Dend.dend_stat[:,:2]  = Dend.dend_stat[:,:2] - np.array([self.SimVars.yLims[0],self.SimVars.xLims[0]])
            np.save(Dend_Stat_Dir, Dend.dend_lumin)
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
            S.points = R.poly.xy.tolist()[:-1]
            ROIline = Line2D(
                np.array(R.poly.xy.tolist())[:,0],
                np.array(R.poly.xy.tolist())[:, 1],
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

    def get_puncta(self):
        """Retrieves and displays puncta based on the current slider values.

        This method shows puncta, adds commands, sets status messages, and performs puncta detection
        using the current slider values. It calculates somatic and dendritic punctas and updates
        the punctas list. Finally, it displays the puncta, updates the status message accordingly,
        and returns None.
        """
        self.show_stuff_coll(["Puncta"])
        self.add_commands(["Puncta"])
        self.set_status_message.setText(self.status_msg["11"])
        QCoreApplication.processEvents()
        self.set_status_message.repaint()
        somas = self.get_soma_polygons()


        soma_thresh = self.puncta_soma_slider.value()/100.0
        dend_thresh = self.puncta_dend_slider.value()/100.0

        PD = PunctaDetection(self.SimVars,self.tiff_Arr,somas,self.DendArr,dend_thresh,soma_thresh)
        somatic_punctas, dendritic_punctas = PD.GetPunctas()
        self.punctas = [somatic_punctas,dendritic_punctas]
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
        try:
            self.update_plot_handle(
                self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :]
            )
        except:
            pass
        self.plot_puncta(self.punctas[0][int(self.timestep_slider.value())][int(self.channel_slider.value())],"soma")
        self.plot_puncta(self.punctas[1][int(self.timestep_slider.value())][int(self.channel_slider.value())],"dendrite")

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
                c = plt.Circle((puncta_x, puncta_y), puncta_r, color="r", linewidth=0.5, fill=False)
            else:
                c = plt.Circle((puncta_x, puncta_y), puncta_r, color="y", linewidth=0.5, fill=False)
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
        for i,R in enumerate(self.roi_interactor_list):
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                self.SpineArr[i].points = R.poly.xy.tolist()[:-1]
                if(self.SpineArr[i].type==2):
                    soma_dict.append(np.asarray(self.SpineArr[i].points))
            else:
                self.SpineArr[i].points[self.actual_timestep] = R.poly.xy.tolist()[:-1]
                if(self.SpineArr[i].type==2):
                    soma_dict.append(np.asarray(self.SpineArr[i].points[0]))
        return soma_dict

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
            try:
                Dend_Dir = self.SimVars.Dir + "Dendrite/"
                if os.path.exists(Dend_Dir):
                    shutil.rmtree(Dend_Dir)
                os.mkdir(path=Dend_Dir)
                for i,Dend in enumerate(self.DendArr):
                    self.dend_measure(Dend,i,Dend_Dir)
                DendSave_csv(Dend_Dir,self.DendArr)
                DendSave_json(Dend_Dir,self.DendArr,self.tiff_Arr,self.SimVars.Snapshots,self.SimVars.Channels,self.SimVars.Unit)
            except Exception as e:
                if DevMode: print(e)
                SaveFlag[0] = False
                pass
        else:
            SaveFlag[0] = False

        if(len(self.SpineArr)>0):
            try:
                Spine_Dir = self.SimVars.Dir + "Spine/"
                if os.path.exists(Spine_Dir):
                    for file_name in os.listdir(Spine_Dir):
                        file_path = os.path.join(Spine_Dir, file_name)
                        
                        # check if the file is the one to keep
                        if ((file_name.startswith('Synapse_a') and self.SimVars.Mode=="Luminosity")
                            or (file_name.startswith('Synapse_l') and self.SimVars.Mode=="Area")):
                            continue  # skip the file if it's the one to keep
                        # delete the file if it's not the one to keep
                        os.remove(file_path)
                else:
                    os.mkdir(path=Spine_Dir)

                T = np.argsort([s.distance for s in self.SpineArr])

                orderedSpineArr = [self.SpineArr[t] for t in T]
                #save spine masks
                for i,t in  enumerate(T):
                    R = self.roi_interactor_list[t]
                    Spine_Mask_Dir = Spine_Dir + "Mask_" + str(i) + ".png"
                    xperts = R.getPolyXYs()
                    mask = np.zeros_like(self.tiff_Arr[0,0])
                    c = np.clip(xperts[:, 0],0,self.tiff_Arr.shape[-2]-1)
                    r = np.clip(xperts[:, 1],0,self.tiff_Arr.shape[-1]-1)
                    rr, cc = polygon(r, c)
                    mask[rr, cc] = 255
                    cv.imwrite(Spine_Mask_Dir, mask)
                nSnaps = self.number_timesteps if self.SimVars.multitime_flag else 1
                nChans = self.number_channels if self.SimVars.multiwindow_flag else 1
                if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                    SaveSynDict(orderedSpineArr, Spine_Dir, "Luminosity",[self.SimVars.yLims,self.SimVars.xLims])
                    SpineSave_csv(Spine_Dir,orderedSpineArr,nChans,nSnaps,'Luminosity',[self.SimVars.yLims,self.SimVars.xLims])
                else:
                    SaveSynDict(orderedSpineArr, Spine_Dir, self.SimVars.Mode,[self.SimVars.yLims,self.SimVars.xLims])
                    SpineSave_csv(Spine_Dir,orderedSpineArr,nChans,nSnaps,self.SimVars.Mode,[self.SimVars.yLims,self.SimVars.xLims])
            except Exception as e:
               if DevMode: print(e)
               SaveFlag[1] = False
               pass
        else:
            SaveFlag[1] = False
        if(len(self.punctas)>0):
            try:
                puncta_Dir = self.SimVars.Dir + "/Puncta/"
                os.makedirs(puncta_Dir, exist_ok=True)

                if os.path.exists(puncta_Dir):
                    shutil.rmtree(puncta_Dir)
                os.makedirs(puncta_Dir,exist_ok=True)
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
        else:
            Text = ""
            if(SaveFlag[0]):
                Text += "Dendrite saved properly, "
            if(SaveFlag[1]):
                Text += "Synapses saved properly, "
            if(SaveFlag[2]):
                Text += "Punctas saved properly"
            self.set_status_message.setText(Text)
        self.PlotSyn()
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
    
        Settings_File = self.SimVars.Dir + "Settings.txt"

        if os.path.exists(Settings_File):
            os.remove(Settings_File)

        file = open(Settings_File, "w")
        values = [("multi-time",self.SimVars.multitime_flag),
                 ("resolution",self.SimVars.Unit),
                ("Image threshold",self.thresh_slider.value()),
                ("Dendritic width",self.neighbour_slider.value()),
                ("Dend. width multiplier",self.dend_width_mult_slider.value()),
                ("ML Confidence",self.ml_confidence_slider.value()),
                ("ROI Tolerance",self.tolerance_slider.value()),
                ("ROI Sigma",self.sigma_slider.value()),
                ("Dendritic puncta threshold",self.puncta_dend_slider.value()),
                ("Somatic puncta threshold",self.puncta_soma_slider.value()),
                ("MLLocation",self.NN_path)
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
                    for i,t in enumerate(T):
                        S = self.SpineArr[t]
                        xy = np.array(S.points[i])
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

                fig.savefig(self.SimVars.Dir+'ROIs_'+str(i)+'.png')

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

            fig.savefig(self.SimVars.Dir+'ROIs.png')

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
        for index, (point,flag) in enumerate(zip(points,flags)):
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                xpert, shift, bgloc,closest_Dend = FindShape(
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
                if(not self.local_shift):
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas,pol))
                else:
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, 
                        pol,point,shift,self.actual_timestep,self.SimVars.Snapshots))
                self.SpineArr.append(Synapse(list(point),list(bgloc),pts=xpert,
                    shift=shift,channel=self.actual_channel,Syntype=flag,closest_Dend=closest_Dend))
            else:
                self.SpineArr.append(Synapse(list(point),[],pts=[],shift=[],channel=self.actual_channel,Syntype=flag))
                
                tiff_Arr_small = tf[:,
                                max(point[1] - 50, 0) : min(point[1] + 50, tf.shape[-2]),
                                max(point[0] - 50, 0) : min(point[0] + 50, tf.shape[-1]),
                            ]
                self.SpineArr[-1].shift = SpineShift(tiff_Arr_small).T.astype(int).tolist()
                for i in range(self.SimVars.Snapshots):
                    xpert, shift, radloc,closest_Dend = FindShape(
                        tf[i],
                        np.array(self.SpineArr[-1].location),
                        medial_axis_Arr,
                        points,
                        mean,
                        True,
                        sigma=self.sigma_val,
                        tol  = self.tol_val,
                        SpineShift_flag = self.local_shift
                    )
                    self.SpineArr[-1].points.append(xpert)
                    self.SpineArr[-1].closest_Dend = closest_Dend
                polygon = np.array(self.SpineArr[-1].points[self.actual_timestep])
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                if(not self.local_shift):
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas,pol))
                else:
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, 
                        pol,point,shift,self.actual_timestep,self.SimVars.Snapshots))

            self.set_status_message.setText(self.set_status_message.text()+'.')
            QCoreApplication.processEvents()
            self.set_status_message.repaint()

        if(self.SimVars.Mode=="Luminosity"): MakeButtonActive(self.spine_bg_button)
        MakeButtonActive(self.measure_spine_button)
        MakeButtonActive(self.measure_puncta_button)
        self.spine_button_ROI.setChecked(False)

        self.set_status_message.setText(self.status_msg["9"])

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
        for S in self.SpineArr:
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                xpert = S.points
                polygon = np.array(xpert)
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                if(S.shift is None):
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas,pol))
                else:
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, 
                        pol,S.location,S.shift,self.actual_timestep,self.SimVars.Snapshots))
                    self.local_shift = True
                    self.local_shift_check.blockSignals(True)
                    self.local_shift_check.setChecked(True)
                    self.local_shift_check.blockSignals(False)

            else:
                polygon = np.array(S.points[self.actual_timestep])
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                if(S.shift is None):
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas,pol))
                else:
                    self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, 
                        pol,S.location,S.shift,self.actual_timestep,self.SimVars.Snapshots))
                    self.local_shift = True
                    self.local_shift_check.blockSignals(True)
                    self.local_shift_check.setChecked(True)
                    self.local_shift_check.blockSignals(False)
 
        points = self.spine_marker.points.astype(int)[[list(sp) not in [S.location for S in self.SpineArr] for sp in self.spine_marker.points]]
        flags  = self.spine_marker.flags.astype(int)[[list(sp) not in [S.location for S in self.SpineArr] for sp in self.spine_marker.points]]

        if(not self.SimVars.multitime_flag):
            tf = self.tiff_Arr[self.actual_timestep,self.actual_channel]
        else:
            tf = self.tiff_Arr[:,self.actual_channel]

        for index, (point,flag) in enumerate(zip(points,flags)):
            if(self.SimVars.Mode=="Luminosity" or not self.SimVars.multitime_flag):
                xpert, shift, bgloc,closest_Dend = FindShape(
                    tf,
                    point,
                    medial_axis_Arr,
                    points,
                    mean,
                    True,
                    sigma=self.sigma_val,
                    tol  = self.tol_val,
                    SpineShift_flag=self.local_shift
                )
                polygon = np.array(xpert)
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, pol,shift))
                self.SpineArr.append(Synapse(list(point),list(bgloc),pts=xpert,
                    shift=shift,channel=self.actual_channel,Syntype=flag,closest_Dend=closest_Dend))
            else:
                self.SpineArr.append(Synapse(list(point),[],pts=[],shift=[],channel=self.actual_channel,Syntype=flag))
                
                tiff_Arr_small = tf[:,
                                max(point[1] - 50, 0) : min(point[1] + 50, tf.shape[-2]),
                                max(point[0] - 50, 0) : min(point[0] + 50, tf.shape[-1]),
                            ]
                self.SpineArr[-1].shift = SpineShift(tiff_Arr_small).T.astype(int).tolist()
                for i in range(self.SimVars.Snapshots):
                    xpert, _, radloc,closest_Dend = FindShape(
                        tf[i],
                        np.array(self.SpineArr[-1].location),
                        medial_axis_Arr,
                        points,
                        mean,
                        True,
                        sigma=self.sigma_val,
                        tol  = self.tol_val
                    )
                    self.SpineArr[-1].points.append(xpert)
                    self.SpineArr[-1].closest_Dend = closest_Dend
                polygon = np.array(self.SpineArr[-1].points[self.actual_timestep])
                pol = Polygon(polygon, fill=False, closed=True, animated=True)
                self.mpl.axes.add_patch(pol)
                self.roi_interactor_list.append(RoiInteractor(self.mpl.axes, self.mpl.canvas, pol,[[0,0]*self.SimVars.Snapshots]))

            self.set_status_message.setText(self.set_status_message.text()+'.')
            QCoreApplication.processEvents()
            self.set_status_message.repaint()
        if(self.SimVars.Mode=="Luminosity"): MakeButtonActive(self.spine_bg_button)
        MakeButtonActive(self.measure_spine_button)
        MakeButtonActive(self.measure_puncta_button)

        self.old_ROI_button.setChecked(False)

        self.set_status_message.setText(self.status_msg["9"])

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

        self.show_stuff_coll([])
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

    
    def on_projection_changed(self):

        """Handles the change of projection selection.

        The method is triggered when the user changes the selection of the projection option.
        It retrieves the new projection value, handles the editing finish for channel 0, and
        handles the editing finish for channel 1, if required.

        Returns:
            None
        """

        new_proj = self.projection.currentText()
        self.handle_editing_finished(0)
        self.handle_editing_finished(1,True)
        
    
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
        
    
    def handle_editing_finished(self,indx,CallTwice=False):
        """Handles the behaviour of the GUI after the folder and resolution are chosen.

        Args:
            indx (int): the flag to choose between the two input fields
            CallTwice (bool, optional): Indicates if the method is called twice. Defaults to False.

        Returns:
            None
        """

        Dir = self.folderpath
        cell = "cell_"+self.cell.text()
        Mode = self.analyze.currentText()
        self.multiwindow_check.setChecked(True)
        multwin = self.multiwindow_check.isChecked()

        res = self.res.text()
        if(res==""):
            res = 0
        projection = self.projection.currentText()
        instance = self

        if(indx==0):

            try:
                self.neighbour_slider.disconnect()
                self.thresh_slider.disconnect()
                self.ml_confidence_slider.disconnect()
                self.sigma_slider.disconnect()
                self.tolerance_slider.disconnect()
                self.puncta_dend_slider.disconnect()
                self.puncta_soma_slider.disconnect()
                self.dend_width_mult_slider.disconnect()
            except Exception as e:
                pass
            self.SimVars = Simulation(res, 0, Dir + "/" + cell + "/", 1, Mode, projection, instance)
            self.SimVars.multitime_flag = self.multitime_check.isChecked()
            self.SimVars.multiwindow_flag = self.multiwindow_check.isChecked()
            try:
                self.tiff_Arr, self.SimVars.Times, meta_data, scale = GetTiffData(None, float(res), self.SimVars.z_type, self.SimVars.Dir,
                                                                    Channels=multwin)
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
            if os.path.exists(Settings_File):
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
                            else:
                                self.NN = False
                                self.button_set_NN.setText("Set NN! (saved can't be found)")
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
                            elif(key=="resolution"):
                                scale = float(value)
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
                                elif(key=="Dendritic puncta threshold"):
                                    self.puncta_soma_slider.setValue(value)
                                    self.puncta_soma_counter.setText(str(value))
                                elif(key=="Dend. width multiplier"):
                                    self.dend_width_mult_slider.setValue(value)
                                    dend_factor = "{:.1f}".format(self.get_actual_multiple_factor())
                                    self.dend_width_mult_counter.setText(dend_factor)
                except Exception as e:
                    self.set_status_message.setText('There was a problem with the settings file')
                    if DevMode: print(e)

            self.SimVars.model = self.NN_path
            self.neighbour_slider.valueChanged.connect(self.dendritic_width_eval)
            self.thresh_slider.valueChanged.connect(self.dend_thresh)
            self.ml_confidence_slider.valueChanged.connect(self.thresh_NN)
            self.sigma_slider.valueChanged.connect(self.spine_tolerance_sigma)
            self.tolerance_slider.valueChanged.connect(self.spine_tolerance_sigma)
            self.puncta_soma_slider.valueChanged.connect(self.soma_threshold_slider_update)
            self.puncta_dend_slider.valueChanged.connect(self.dend_threshold_slider_update)
            self.multitime_check.stateChanged.connect(lambda state: self.check_changed(state,1))
            self.multiwindow_check.stateChanged.connect(lambda state: self.check_changed(state,0))
            self.dend_width_mult_slider.valueChanged.connect((self.dendritic_width_eval))
            self.SimVars.Unit = scale
            # Get shifting of snapshots
            if (self.SimVars.Snapshots > 1):
                self.tiff_Arr = GetTiffShift(self.tiff_Arr, self.SimVars)

                # Get Background values
            cArr_m = np.zeros_like(self.tiff_Arr[0, :, :, :])
            for i in range(self.SimVars.Channels):
                cArr_m[i, :, :] = canny(self.tiff_Arr[:, i, :, :].max(axis=0), sigma=1)
                self.SimVars.bgmean[:, i] = Measure_BG(self.tiff_Arr[:, i, :, :], self.SimVars.Snapshots, self.SimVars.z_type)

            self.mpl.image = self.tiff_Arr[0,0]

            self.multiwindow_check.setEnabled(True)
            self.multitime_check.setEnabled(True)
            self.projection.setEnabled(True)
            self.analyze.setEnabled(True)
            if(scale>0):
                self.CheckOldDend()
                self.res.setText("%.3f" % scale)
                MakeButtonActive(self.medial_axis_path_button)
        if(indx==1):
            if(not CallTwice):
                self.clear_stuff(True)
            self.SimVars.Unit = float(self.res.text())
            MakeButtonActive(self.medial_axis_path_button)
            self.CheckOldDend()
        
    def get_actual_multiple_factor(self):
        return 0.05*self.dend_width_mult_slider.value()

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
                self.DendArr.append(Dend)
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

            if((os.path.isfile(SpineDir+'Synapse_l.json') and self.SimVars.Mode=="Luminosity") or
                (os.path.isfile(SpineDir+'Synapse_a.json') and self.SimVars.Mode=="Area" and self.SimVars.multitime_flag)):
                MakeButtonActive(self.old_ROI_button)
            self.set_status_message.setText(self.status_msg["10"])
            self.mpl.canvas.draw()

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

        MakeButtonActive(self.spine_button_ROI)

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
        It then displays a spine marker based on the spine points with scores above the threshold.

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
        
    
    def dendritic_width_eval(self) -> None:
        """
        function that performs the dendritic width calculation
        when the button is pressed
        Returns: None

        """
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
            D.actual_channel = self.actual_channel
            D.actual_timestep= self.actual_timestep
            D.set_surface_contours(
                max_neighbours=self.neighbour_slider.value(), sigma=10, width_factor=dend_factor
            )
            dend_surface = D.get_dendritic_surface_matrix()
            dend_cont = D.get_contours()
            polygon = np.array(dend_cont[0][:, 0, :])
            pol = Polygon(dend_cont[0][:, 0, :], fill=False, closed=True,color='r')
            self.mpl.axes.add_patch(pol)
            self.mpl.canvas.draw()

        MakeButtonActive(self.save_button)
        MakeButtonActive(self.measure_puncta_button)
        self.dendritic_width_button.setChecked(False)
    
    def medial_axis_eval_handle(self) -> None:
        """
        performs the medial axis calcultation
        Returns: None
        """
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
        self.DendMeasure= medial_axis_eval(self.SimVars,self.tiff_Arr,self.DendArr,self)
        self.medial_axis_path_button.setChecked(False)
        
    def get_path(self) -> None:
        """
        opens a dialog field where you can select the folder
        Returns: None

        """
        path = QFileDialog.getExistingDirectory(self, "Select Folder!")
        if(path):
            self.folderpath = path
            self.folderpath_label.setText(str(self.folderpath))
            self.set_status_message.setText(self.status_msg["1"])
            self.cell.setEnabled(True)
            self.res.setEnabled(True)
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

        mean = np.mean(self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :])
        max = np.max(self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :])
        self.thresh_slider.blockSignals(True)
        self.thresh_slider.setMinimum(int(mean))
        self.thresh_slider.setMaximum(int(max))
        self.thresh_slider.setValue(int(mean))
        self.thresh_slider.blockSignals(False)

        self.update_plot_handle(
            self.tiff_Arr[self.actual_timestep, self.actual_channel, :, :])
        if(self.PunctaCalc):
            self.display_puncta()

        self.mpl.canvas.setFocus()
    
    def SaveROIstoSpine(self):
        try:
            if(self.SimVars.Mode=="Luminosity"):
                for i,R in enumerate(self.roi_interactor_list):
                    R.poly.xy = R.poly.xy - R.shift[R.Snapshot]
                    self.SpineArr[i].points = (R.poly.xy)[:-1].tolist()
            else:
                for i,R in enumerate(self.roi_interactor_list):
                    if(self.local_shift):
                        R.poly.xy = R.poly.xy - R.shift[R.Snapshot]
                        self.SpineArr[i].points[R.Snapshot] = (R.poly.xy)[:-1].tolist()
                    else:
                        self.SpineArr[i].points[R.Snapshot] = (R.poly.xy)[:-1].tolist()
        except Exception as e:
            print(e)
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
                for i,R in enumerate(self.roi_interactor_list):
                    R.poly.xy = R.poly.xy - R.shift[R.Snapshot]
                    self.SpineArr[i].points = (R.poly.xy)[:-1].tolist()
                    self.SpineArr[i].shift[R.Snapshot] = R.shift[R.Snapshot]
                    R.Snapshot = self.actual_timestep
                    R.poly.xy = R.poly.xy + R.shift[R.Snapshot]
                    R.line.set_data(zip(*R.poly.xy))
                    R.line_centre.set_data([R.OgLoc[0]+R.shift[R.Snapshot][0],R.OgLoc[1]+R.shift[R.Snapshot][1]])
                    R.loc = [R.OgLoc[0]+R.shift[R.Snapshot][0],R.OgLoc[1]+R.shift[R.Snapshot][1]]
                    if(self.actual_timestep>0):
                        R.line_centre.set_color('r')
                        R.line_centre.set_markerfacecolor('k')
                    else:
                        R.line_centre.set_color('gray')
                        R.line_centre.set_markerfacecolor('gray')

            else:
                for i,R in enumerate(self.roi_interactor_list):
                    if(self.local_shift):
                        R.poly.xy = R.poly.xy - R.shift[R.Snapshot]
                        self.SpineArr[i].points[R.Snapshot] = (R.poly.xy)[:-1].tolist()
                        self.SpineArr[i].shift[R.Snapshot] = R.shift[R.Snapshot]
                        R.Snapshot = self.actual_timestep
                        newDat = np.array(self.SpineArr[i].points[R.Snapshot])+np.array(R.shift[R.Snapshot])
                        R.poly.xy = newDat
                        R.line.set_data(newDat[:,0],newDat[:,1])
                        R.line_centre.set_data([R.OgLoc[0]+R.shift[R.Snapshot][0],R.OgLoc[1]+R.shift[R.Snapshot][1]])
                        R.loc = [R.OgLoc[0]+R.shift[R.Snapshot][0],R.OgLoc[1]+R.shift[R.Snapshot][1]]
                        R.points =  np.array(R.poly.xy)-np.array(R.loc)
                        if(self.actual_timestep>0):
                            R.line_centre.set_color('r')
                            R.line_centre.set_markerfacecolor('k')
                        else:
                            R.line_centre.set_color('gray')
                            R.line_centre.set_markerfacecolor('gray')
                    else:
                        self.SpineArr[i].points[R.Snapshot] = (R.poly.xy)[:-1].tolist()
                        R.Snapshot = self.actual_timestep
                        newDat = np.array(self.SpineArr[i].points[R.Snapshot])
                        R.poly.xy = newDat
                        R.line.set_data(newDat[:,0],newDat[:,1])
        except Exception as e:
           # Print the error message associated with the exception
           pass
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
        self.hide_stuff([self.thresh_slider, self.thresh_label])
        self.hide_stuff([self.neighbour_counter,self.neighbour_slider,self.neighbour_label])
        self.hide_stuff([self.sigma_label,self.sigma_counter,self.sigma_slider,
            self.tolerance_label,self.tolerance_counter,self.tolerance_slider])
        self.hide_stuff([self.ml_confidence_label,self.ml_confidence_slider,self.confidence_counter ])
        self.hide_stuff([self.dend_width_mult_label, self.dend_width_mult_slider, self.dend_width_mult_counter])
        self.hide_stuff([self.local_shift_check])

        for Name in Names:
            if(Name=="Puncta"):
                self.show_stuff([self.puncta_dend_label,self.puncta_dend_slider,self.puncta_dend_counter])
                self.show_stuff([self.puncta_soma_label,self.puncta_soma_slider,self.puncta_soma_counter])
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
            elif(flag==2):
                self.local_shift = True
                self.spine_ROI_eval()
        else:
            if(flag==0):
                self.SimVars.multiwindow_flag = False
                self.hide_stuff([self.channel_label,self.channel_slider,self.channel_counter])
            elif(flag==1):
                self.SimVars.multitime_flag = False
                self.hide_stuff([self.timestep_label,self.timestep_slider,self.timestep_counter])
            elif(flag==2):
                self.local_shift = False
                self.spine_ROI_eval()

@handle_exceptions
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
        MakeButtonActive(self.sourcepath_button)
        self.grid.addWidget(self.sourcepath_button, 0, 0)
        self.sourcepath_label = QLineEdit(self)
        self.sourcepath_label.setReadOnly(True)
        self.sourcepath_label.setText(str(self.sourcepath))
        self.grid.addWidget(self.sourcepath_label, 0, 1)

        self.targetpath_button = QPushButton(self)
        self.targetpath_button.setText("Select target path!")
        MakeButtonInActive(self.targetpath_button)
        self.targetpath_button.clicked.connect(self.get_target)
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
            self.set_status_message.setText("Your source had no lsm or tif files")
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
    """Class that defines the directory structure window"""

    def __init__(self):
        super().__init__()

        self.title = "Tutorial Window"
        self.left = 100
        self.top = 100
        self.width = 200
        self.height = 400
        self.initUI()

        self.foldurl = 'https://www.youtube.com/watch?v=3GOStVqGbA0'

        self.dendurl = 'https://www.youtube.com/watch?v=3GOStVqGbA0'

        self.spineurl = 'https://www.youtube.com/watch?v=3GOStVqGbA0'

        self.punctaurl = 'https://www.youtube.com/watch?v=3GOStVqGbA0'

        self.filesurl = 'https://www.youtube.com/watch?v=3GOStVqGbA0'

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

        self.DendriteTutorial_button = QPushButton(self)
        self.DendriteTutorial_button.setText("Analysing a dendrite")
        MakeButtonActive(self.DendriteTutorial_button)
        self.DendriteTutorial_button.clicked.connect((lambda: self.LoadURL(self.dendurl)))
        self.grid.addWidget(self.DendriteTutorial_button, 1, 0, 1, 1)

        self.SpineTutorial_button = QPushButton(self)
        self.SpineTutorial_button.setText("Analysing spines")
        MakeButtonActive(self.SpineTutorial_button)
        self.SpineTutorial_button.clicked.connect((lambda: self.LoadURL(self.spineurl)))
        self.grid.addWidget(self.SpineTutorial_button, 2, 0, 1, 1)

        self.PunctaTutorial_button = QPushButton(self)
        self.PunctaTutorial_button.setText("Analysing puncta")
        MakeButtonActive(self.PunctaTutorial_button)
        self.PunctaTutorial_button.clicked.connect((lambda: self.LoadURL(self.punctaurl)))
        self.grid.addWidget(self.PunctaTutorial_button, 3, 0, 1, 1)

        self.FileTutorial_button = QPushButton(self)
        self.FileTutorial_button.setText("The file structure")
        MakeButtonActive(self.FileTutorial_button)
        self.FileTutorial_button.clicked.connect((lambda: self.LoadURL(self.fileurl)))
        self.grid.addWidget(self.FileTutorial_button, 4, 0, 1, 1)

    def LoadURL(self,url):

        wb.open(url)

        self.FolderTutorial_button.setChecked(False)
        self.DendriteTutorial_button.setChecked(False)
        self.SpineTutorial_button.setChecked(False)
        self.PunctaTutorial_button.setChecked(False)
        self.FileTutorial_button.setChecked(False)


class MainWindow(QWidget):
    """
    class that makes the main window
    """

    def __init__(self):
        super().__init__()
        self.title = "Dendritic Spine Tool"
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
        self.setWindowIcon(QIcon(QPixmap("brain.png")))
        self.grid = QGridLayout(self)

        # headline
        self.headline = QLabel(self)
        self.headline.setTextFormat(Qt.TextFormat.RichText)
        self.headline.setText("The Dendritic Spine Tool <br> <font size='0.1'>v0.6.2-alpha</font>")
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
        relative_address = "dend.jpg"
        # Construct the absolute path of the image file within the package structure
        image_path_in_package = os.path.join(code_dir, relative_address)

        pixmap = QPixmap(image_path_in_package)
        pixmap = pixmap.scaled(900, 600)
        self.image = QLabel(self)
        self.image.setPixmap(pixmap)
        self.grid.addWidget(self.image, 2, 1, 6, 6)

        # read data button
        self.read_data_button = QPushButton(self)
        self.read_data_button.setText("Read Data")
        MakeButtonActive(self.read_data_button,1)
        self.read_data_button.clicked.connect(self.read_data)
        self.grid.addWidget(self.read_data_button, 8, 1, 2, 2)

        # Tutorial button
        self.tutorial_button = QPushButton(self)
        self.tutorial_button.setText("Tutorials")
        MakeButtonActive(self.tutorial_button,1)
        self.tutorial_button.clicked.connect(self.tutorial)
        self.grid.addWidget(self.tutorial_button, 8, 3, 2, 2)

        # Analyze button
        self.generate_button = QPushButton(self)
        self.generate_button.setText("Generate folders")
        MakeButtonActive(self.generate_button,1)
        self.generate_button.clicked.connect(self.generate)
        self.grid.addWidget(self.generate_button, 8, 5, 2, 2)

        #========= multiwindow checkbox ================
        self.DevMode_check = QCheckBox(self)
        self.DevMode_check.setText("Developer Mode")
        self.grid.addWidget(self.DevMode_check, 7, 3, 1, 1)
        self.DevMode_check.stateChanged.connect(lambda state: self.check_changed(state))


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
