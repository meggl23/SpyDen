import sys
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import torchvision.transforms as transforms
import os
import requests

import torch
import torchvision.transforms as transforms
from skimage import feature
from skimage.draw import ellipse

from PyQt5.QtCore import QCoreApplication


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

def MakeButtonActive(button,Flag=0):
    """Sets the given button as active and enables it with custom styles.

    The function sets the given button as active by making it checkable and enabling it.
    It also applies custom styles to the button based on the provided Flag.

    Args:
        button (QPushButton): The button to be made active.
        Flag (int, optional): A flag to determine the custom styles applied to the button.
            If Flag is 0, the button will have a smaller size and font size.
            If Flag is non-zero, the button will have a larger size and font size.
            Defaults to 0.

    Returns:
        None
    """
    if(Flag==0):
        button.setCheckable(True)
        button.setStyleSheet(
            "QPushButton {"
            "background-color: green;"
            "font-family: Courier;"
            "color: white;"
            "border-style: outset;"
            "border-radius: 8px;"
            "border-color: black;"
            "border-width : 2px;"
            "font: bold 15px;"
            "padding: 3px;"
            "}"
            "QPushButton:checked {"
            "background-color: #80c080;"
            "font-family: Courier;"
            "color: white;"
            "border-style: outset;"
            "border-radius: 8px;"
            "border-color: black;"
            "border-width : 2px;"
            "font: bold 15px;"
            "padding: 3px;"
            "}"
        )
    else:
        button.setCheckable(True)
        button.setStyleSheet(
            "QPushButton {"
            "background-color: green;"
            "font-family: Courier;"
            "color: white;"
            "border-style: outset;"
            "border-radius: 10px;"
            "border-width : 2px;"
            "font: bold 30px;"
            "padding: 30px;"
            "border-color: black"
            "}"
            "QPushButton:checked {"
            "background-color: #80c080;"
            "font-family: Courier;"
            "color : white;"
            "border-style: outset;"
            "border-radius: 10px;"
            "border-width : 2px;"
            "font: bold 30px;"
            "padding: 30px;"
            "border-color: black"
            "}"
        )

    button.setEnabled(True)

    

def MakeButtonInActive(button):

    button.setStyleSheet(
        "QPushButton {"
        "background-color: gray;"
        "font-family: Courier;"
        "color: white;"
        "border-style: outset;"
        "border-radius: 8px;"
        "border-color: white;"
        "border-width: 0px;"
        "font: bold 15px;"
        "padding: 3px;"
        "}"
    )

    button.setEnabled(False)
    

def download_model(model_url,save_path,Simvars):


    response = requests.get(model_url,stream = True)
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    Simvars.frame.set_status_message.setText("Downloading:")
    try:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = int((downloaded / total_size) * 100) if total_size else 0
                    # Update the text field with the current progress
                    Simvars.frame.set_status_message.setText(f"Downloading: {percent}%")
                    # Process pending GUI events so the text field updates
                    QCoreApplication.processEvents()
    except Exception as e:
        print(e)
        os.remove(save_path)

def load_model(Simvars):

    # Create the full path
    Simvars.frame.set_status_message.setText("Loading model")

    folder_path = os.path.expanduser("~")
    file_name = 'model.pth'
    model_path = os.path.join(folder_path, file_name)

    if not os.path.exists(model_path):
        download_model(Simvars.ML_URL,model_path,Simvars)
    model = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False)
    # Additional model setup, if needed
    try:
        os.remove(model_path)
    except:
        pass
    Simvars.model = model


def RunNN(Simvars, DendArr, tiff_Arr):
    """
    Runs a neural network model on the provided data and returns predicted points and scores.

    Args:
        Simvars: Simvars object.
        DendArr (numpy.ndarray): Array of dendrite points.
        tiff_Arr (numpy.ndarray): Array of TIFF data.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Tuple containing predicted points and scores.

    """
    model = torch.load(Simvars.model, map_location=torch.device("cpu"))
    model.eval()
    BoxsList = []
    ScoreList = []
    offset = [0]
    Training_x = []
    Training_y = []
    d = DendArr
    len_x = len(tiff_Arr[0, :])
    len_y = len(tiff_Arr[ :, 0])

    tiff_Arr = tiff_Arr.reshape(1, len_y, len_x)
    xmin, xmax = max(min(d[:, 1]) - 50, 0), min(max(d[:, 1]) + 50, tiff_Arr.shape[-2])
    ymin, ymax = max(min(d[:, 0]) - 50, 0), min(max(d[:, 0]) + 50, tiff_Arr.shape[-1])
    im = tiff_Arr[None, 0, int(xmin) : int(xmax), int(ymin) : int(ymax)]
    im = np.repeat(im, 3, axis=0)

    im = im/np.max(im)
    im = data_transforms["val"](np.moveaxis(im, 0, -1).astype(np.single))[None, :, :, :]


    

    testOut = model(im)
    sBoxsList = testOut[0]["boxes"].detach().numpy()
    sScoreList = testOut[0]["scores"].detach().numpy()

    i = 0
    tBoxsList = np.copy(sBoxsList)
    tScoreList = np.copy(sScoreList)
    while i < len(tBoxsList):
        poplist = []
        for j, b in enumerate(tBoxsList):
            if iou(tBoxsList[i], b) > 0 and iou(tBoxsList[i], b) < 1:
                poplist.append(j)
        tBoxsList = np.delete(tBoxsList, poplist, axis=0)
        tScoreList = np.delete(tScoreList, poplist)
        i = i + 1
    pPoints = []

    score = tScoreList
    for b in tBoxsList:
        pPoints.append([(b[0] + b[2]) / 2 + ymin, (b[1] + b[3]) / 2 + xmin])
    pPoints = np.array(pPoints)
    return pPoints, score


def getWidthnew(img, all_ps, sigma, max_neighbours, width_factor = 1):
    """
    Calculates the width of the dendrite along the provided points.

    Args:
        img (numpy.ndarray): Image array.
        all_ps (numpy.ndarray): Array of points.
        sigma (float): Sigma value for the Canny edge detection.
        max_neighbours (int): Maximum number of neighboring points to consider.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the width array and degrees array.

    """


    edges1 = feature.canny(img, sigma=sigma)


    width_arr = np.zeros(all_ps.shape[0])
    degrees = np.zeros(all_ps.shape[0])
    for dxd, d in enumerate(all_ps[1:]):
        u_vector = GetPerpendicularVector(all_ps[dxd], d)
        if angle_between2(np.array([0, 1]), u_vector) > 180:
            degrees[dxd] = -1 * angle_between(np.array([0, 1]), u_vector)
        else:
            degrees[dxd] = angle_between(np.array([0, 1]), u_vector)
        starting_width = 1

        width_arr[dxd] = SearchWidth(
            img,
            edges1,
            all_ps[dxd],
            degrees[dxd],
            starting_width,
            0,
            max_neighbours=max_neighbours,
        )
    ########################## try to remove abrupt changes
    for i in range(len(width_arr) - 1):
        if width_arr[i + 1] > width_arr[i] + max_neighbours:
            width_arr[i + 1] = width_arr[i]
    try:
        width_arr[-1] = width_arr[-2]
    except Exception as e:
       # Print the error message associated with the exception
        pass
    ##########################
    reverted = np.flip(width_arr, axis=0)
    for i in range(len(reverted) - 1):
        if reverted[i + 1] > reverted[i] + max_neighbours:
            reverted[i + 1] = reverted[i]
    width_arr = np.flip(reverted, axis=0)

    return width_arr*width_factor, degrees


def SearchWidth(img, edges, p, degree, width, count, max_neighbours):
    """
    Recursively searches for the width of a dendrite at a specific point.

    Args:
        img (numpy.ndarray): Image array.
        edges (numpy.ndarray): Array of edge points.
        p (Tuple[float, float]): Point coordinates.
        degree (float): Degree value.
        width (float): Initial width value.
        count (int): Count value to control recursion.
        max_neighbours (int): Maximum number of neighboring points to consider.

    Returns:
        float: Width value.

    """

    rr, cc = ellipse(p[1], p[0], width, 1, rotation=degree, shape=img.shape)
    iix = np.where(edges[rr, cc] == True)[0]

    if iix.shape[0] > 1 or count > 30:
        return width
    next_width = width * 1.2
    count += 1
    return SearchWidth(
        img, edges, p, degree, next_width, count, max_neighbours=max_neighbours
    )

class Simulation:

    """Class that holds the parameters associated with the simulation"""

    def __init__(self, Unit, bgmean, Dir, Snapshots, Mode, z_type, frame=None):
        self.Pic_Type = "MAX"
        self.Unit = 0.06589
        self.bgmean = []
        self.Dir = Dir
        self.SomaSim = False
        self.Snapshots = Snapshots
        self.MinDirCum = []

        self.ML_URL = os.environ.get('MODEL_URL', 'https://gin.g-node.org/CompNeuroNetworks/SpyDenTrainedNetwork/raw/198adbc3017080986f5401dc4046849bad54709e/SynapseMLModel')
        self.model = "SynapseMLModel"

        self.Times = []

        self.SingleClick = True
        self.frame = frame

        self.GetNumpyFunc(z_type)

        self.Mode = Mode

        self.multitime_flag = False
        self.multiwindow_flag = True

        self.xLims = []
        self.yLims = []

    def GetNumpyFunc(self,z_type):
        if z_type == "Sum":
            self.z_type = np.sum
        elif z_type == "Max":
            self.z_type = np.max
        elif z_type == "Min":
            self.z_type = np.min
        elif z_type == "Mean":
            self.z_type = np.mean
        elif z_type == "Median":
            self.z_type = np.median
        elif z_type == "Std":
            self.z_type = np.std
        else:
            self.z_type = None

                
def dist(x, y):
    """
    Return the distance between two points.
    """
    d = x - y
    return np.sqrt(np.dot(d, d))


def dist_point_to_segment(p, s0, s1):
    """
    Get the distance of a point to a segment.
      *p*, *s0*, *s1* are *xy* sequences
    This algorithm from
    http://www.geomalgorithms.com/algorithms.html
    """
    v = s1 - s0
    w = p - s0
    c1 = np.dot(w, v)
    if c1 <= 0:
        return dist(p, s0)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return dist(p, s1)
    b = c1 / c2
    pb = s0 + b * v
    return dist(p, pb)

def segment_intersection(p, r, q, s):
    """
    Compute the intersection of two line segments:
      Segment 1: from point p in direction r
      Segment 2: from point q in direction s

    Solves for parameters t and u in: p + t*r = q + u*s.
    Returns (t, u) if an intersection exists, else returns None.
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    r = np.array(r, dtype=float)
    s = np.array(s, dtype=float)
    
    r_cross_s = np.cross(r, s)
    if np.abs(r_cross_s) < 1e-8:
        return None  # lines are parallel or collinear
    t = np.cross((q - p), s) / r_cross_s
    u = np.cross((q - p), r) / r_cross_s
    return t, u

def find_intersection(polyline, polygon):
    """
    Given a polyline (Nx2 numpy array) and a closed polygon (Mx2 numpy array),
    find the intersection point along the first polyline segment that goes
    from inside the polygon (using matplotlib Path) to outside.
    
    Returns the intersection point and the index of the segment, or (None, None)
    if no intersection is found.
    """
    # Create a Path object from the polygon.
    # If the polygon is not explicitly closed (first != last), Path will close it.
    polyline = np.array(polyline)
    polygon = np.array(polygon)
    if not np.array_equal(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    path = Path(polygon)
    # Determine which polyline points are inside the polygon.
    inside = path.contains_points(polyline)
    
    # Loop over segments of the polyline.
    for i in range(len(polyline) - 1):
        if inside[i] and not inside[i+1]:
            p_inside = polyline[i]
            p_outside = polyline[i+1]
            seg_vec = p_outside - p_inside
            intersections = []
            # Loop through polygon edges.
            # Assume polygon is closed. If not, you can force it with:
            # polygon = np.vstack([polygon, polygon[0]])
            for j in range(len(polygon)-1):
                edge_start = polygon[j]
                edge_end = polygon[j+1]
                edge_vec = edge_end - edge_start
                result = segment_intersection(p_inside, seg_vec, edge_start, edge_vec)
                if result is None:
                    continue
                t, u = result
                if 0 <= t <= 1 and 0 <= u <= 1:
                    ip = p_inside + t * seg_vec
                    intersections.append(ip)
            if intersections:
                intersections = np.array(intersections)
                # Choose the intersection closest to the inside point.
                dists = np.linalg.norm(intersections - p_inside, axis=1)
                idx = np.argmin(dists)
                return intersections[idx], i
    return None, None



def angle_between(v1, v2):

    v1_u = unitVector(v1)
    v2_u = unitVector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between2(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def unitVector(a):
    return a / np.linalg.norm(a)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "same") / w


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def projection(p1, p2, p3):

    """Function that returns the projection of a point onto a line"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    dx, dy = x2 - x1, y2 - y1
    det = dx * dx + dy * dy

    a = (dy * (y3 - y1) + dx * (x3 - x1)) / det

    return x1 + a * dx, y1 + a * dy


def GetPerpendicularVector(a, b):
    """
    Calculates the perpendicular vector between two points.

    Args:
        a (numpy.ndarray): First point coordinates.
        b (numpy.ndarray): Second point coordinates.

    Returns:
        numpy.ndarray: Perpendicular vector.

    """
    u = a - b
    u_p = u / np.linalg.norm(u)

    d_p = np.array([[0, 1], [-1, 0]])

    return d_p.dot(u_p)


def crosslen(a0, a1, b0, b1):

    """Function that returns the projection of a point onto a line"""
    a00, a01 = a0
    a10, a11 = a1
    b00, b01 = b0
    b10, b11 = b1

    lam = ((a01 - b01) * (b10 - b00) - (a00 - b00) * (b11 - b01)) / (
        (a10 - a00) * (b11 - b01) - (a11 - a01) * (b10 - b00)
    )
    mu = ((a00 - b00) + lam * (a10 - a00)) / (b10 - b00)

    if np.isnan(mu) or np.isinf(mu):
        mu = ((a01 - b01) + lam * (a11 - a01)) / (b11 - b01)

    return lam, mu


def averager(arr: np.array, n: int) -> tuple[np.array, np.array]:
    """
    function that computes the average of every n entries in a numpy array
    if len(arr) is not dividible by n the rest is averaged by a smaller amount
    :param arr: array
    :param n:every n elements will be averaged
    :return:averaged numpy array
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"should be 1D np.array given {type(arr)}")

    averages = []
    full_intervals = len(arr) // n

    if len(arr) % n == 0:
        indexes = np.ones(full_intervals + 1) * n
        indexes[0] = 0
        boarders = np.cumsum(indexes).astype(int)

        for i in range(full_intervals):
            res = 1 / n * np.sum(arr[boarders[i] : boarders[i + 1]])
            averages.append(res)
        return np.array(averages), boarders

    else:
        left = len(arr) % n
        indexes = np.ones(full_intervals + 2) * n
        indexes[0], indexes[-1] = 0, left
        boarders = np.cumsum(indexes).astype(int)

        for i in range(full_intervals + 1):
            res = 1 / (indexes[i + 1]) * np.sum(arr[boarders[i] : boarders[i + 1]])
            averages.append(res)
        return np.array(averages), boarders

def curvature_dependent_sampling(arr: np.array, pixels_per_intervall: int, min_points: int = 10) -> np.array:
    """
    Calculate points for a piecewise linear approximation of the medial axis
    based on curvature-dependent sampling. High curvature regions will have more sampling.
    
    :param arr: Curvature per pixel along the dendrite path.
    :param pixels_per_intervall: Interval size for averaging.
    :param min_points: Minimum number of points to guarantee in the final sampling.
    :return: A tuple (sampling, boarders, aver) where:
             - sampling: indices of the sampled points,
             - boarders: interval borders from the averaging,
             - aver: the averaged curvature values.
    """
    aver, boarders = averager(arr, pixels_per_intervall)  # sub-averaging
    res = list(map(curvature_eval, aver))  # evaluate curvature to get number of points per sample

    sampling = np.array([])
    for i in range(len(aver)):
        # For each interval, generate points based on the computed number plus one.
        intervall = np.linspace(boarders[i], boarders[i + 1], res[i] + 1)
        # Exclude the last point to avoid duplicates (except for the final interval)
        sampling = np.append(sampling, intervall[:-2])
    sampling = np.append(sampling, boarders[-1])
    sampling = sampling.astype(int)

    # Ensure a minimum number of points.
    if sampling.size < min_points:
        sampling = np.linspace(boarders[0], boarders[-1], min_points, dtype=int)

    return sampling, boarders, aver


def curvature_eval(x: float) -> int:
    """
    function that calculates the number of sampling points dependent on the mean curvature of an interval
    it is assumed that we have in general not that much curvature on a medial axis path
    therefore there is a threshold where an increasing of curvature will not return more sampling points
    the minimum number of points is 2 per interval
    the estimated parameters rely on observations this could be improved further
    :param x: averaged curvature for a intervall in units Curvature/Pixel
    :return: number of samplings points for given interval
    """

    x = x * 10**5
    if 0 <= x <= 2:
        sampling = 1.5 * x + 2
        return int(sampling)
    if x > 2:
        return 5


def curvature_polygon(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    function calculates tangent and curvature vectors of a polygon and length od vectors
    values have to be ordered
    for plotting tangent vectors x,y  vals have to be reduced by one
    for plotting curvature vectors x,y  vals have to be reduced by one
    :param x: x values
    :param y:y values
    :return: Tx, Ty, Hx, Hy, T, H
    """

    dsx = np.diff(x)
    dsy = np.diff(y)
    ds = np.sqrt(dsx**2 + dsy**2)
    Tx = dsx / ds
    Ty = -dsy / ds
    ds2 = 0.5 * (ds[:-1] + ds[1:])
    Hx = np.diff(Tx) / ds2
    Hy = np.diff(Ty) / ds2
    T = np.sqrt(Tx**2 + Ty**2)
    H = np.sqrt(Hx**2 + Hy**2)

    return Tx, Ty, Hx, Hy, T, H
