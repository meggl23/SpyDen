import queue
from skimage.draw import line

import networkx as nx
from scipy.ndimage import gaussian_filter1d
import torch.nn as nn
import math as mt

from .Utility import *


def FindSoma(mesh, start, end, verbose=False, scale=0.114):

    """
    Input:
            mesh (np.array)  : Bool array of tiff file > some threshold
            start (ints)     : Start of path
            end (ints)       : end of path
            Verbose (Bool)   : Wether you want the program to tell you whats going on
    Output:
            Path along the mesh from start to end

    Function:
            Uses the Breadth-first algorithm to explore the mesh
    """

    count = 0
    add = ""
    maze = mesh
    visited = np.zeros_like(maze)
    direction = np.zeros_like(maze)

    nums = queue.Queue()
    j, i = start
    if valid(maze, visited, start):
        nums.put(start)
    else:
        print("choose valid starting cell")

    endlist = end.tolist()
    visited[j, i] = 1
    while not nums.empty():
        count += 1
        add = nums.get()
        if add.tolist() in endlist or (add == end).all():
            return GetPath(start, add, direction)

        for j, dirc in np.array(
            [
                [[0, 1], 1],
                [[0, -1], 2],
                [[-1, 0], 3],
                [[1, 0], 4],
                [[1, 1], 5],
                [[1, -1], 6],
                [[-1, 1], 7],
                [[-1, -1], 8],
            ]
        ):
            put = add + j
            if valid(maze, visited, put):
                count += 1
                j, i = put
                visited[j, i] = 1
                direction[j, i] = dirc
                nums.put(put)

    return [start[1], start[0]], 0


def valid(maze, visited, moves):
    j, i = moves
    if not (0 <= i < len(maze[0]) and 0 <= j < len(maze)):
        return False
    elif maze[j][i] == 0:
        return False
    elif visited[j][i] == 1:
        return False
    return True


def GetPath(start, end, directions, scale=0.114, shorten=False):

    """
    Input:
            start (ints)     : Start of path
            end (ints)       : end of path
            mesh (np.array)  : Directions on mesh that point towards the start
    Output:
            path and length of shortest path

    Function:
            Propagates the found directions back from the end to the start
    """

    # 4 for Up, 3 for Down, 2 for right and 1 for left
    current = end
    path_arr = [np.array([end[1], end[0]])]
    j, i = current
    length = 0
    fp = []
    sp = []
    while not (current == start).all():
        j, i = current
        if directions[j, i] == 4:
            length += 1
            current = current - [1, 0]
        elif directions[j, i] == 3:
            length += 1
            current = current - [-1, 0]
        elif directions[j, i] == 2:
            length += 1
            current = current - [0, -1]
        elif directions[j, i] == 1:
            length += 1
            current = current - [0, 1]
        elif directions[j, i] == 5:
            length += np.sqrt(2)
            current = current - [1, 1]
        elif directions[j, i] == 6:
            length += np.sqrt(2)
            current = current - [1, -1]
        elif directions[j, i] == 7:
            length += np.sqrt(2)
            current = current - [-1, 1]
        elif directions[j, i] == 8:
            length += np.sqrt(2)
            current = current - [-1, -1]
        else:
            break
            print("there is some error")
        path_arr.append([current[1], current[0]])
    fp, sp = SecondOrdersmoothening(np.asarray(path_arr), np.sqrt(2) / scale)
    return path_arr, length * scale


def SmoothenPath(x, y):
    """
    Smoothes a path defined by x and y coordinates using a simplification algorithm.

    Args:
        x (list): List of x-coordinates of the path points.
        y (list): List of y-coordinates of the path points.

    Returns:
        numpy.ndarray: Numpy array containing the modified coordinates of the smoothed path.

    """
    length = len(x)
    modified_list = [[x[0], y[0]]]
    for i in range(1, length - 1):
        A = np.array([x[i] - x[i - 1], y[i] - y[i - 1]])
        B = np.array([x[i] - x[i + 1], y[i] - y[i + 1]])
        cp = np.cross(A, B)
        if cp != 0:
            modified_list.append([x[i], y[i]])
    modified_list.append([x[-1], y[-1]])
    modified_list = np.asarray(modified_list)
    return modified_list


def SecondOrdersmoothening(orig_path, min_dist):
    """
    Applies second-order smoothing to a path by removing points that are closer than a specified minimum distance.

    Args:
        orig_path (numpy.ndarray): Original path defined by x and y coordinates.
        min_dist (float): Minimum distance threshold.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the modified path after second-order smoothing and the path after first-order smoothing.

    """
    first_order = SmoothenPath(orig_path[:, 0], orig_path[:, 1])
    second_order = [first_order[0]]
    for vdx, v in enumerate(first_order[0:-2]):
        if dist(v, first_order[vdx + 1]) > min_dist:
            second_order.append(first_order[vdx + 1])
    if not (second_order[-1] == first_order[-1]).all():
        second_order.append(first_order[-1])
    return np.asarray(second_order), first_order


def GetAllpointsonPath(xys):
    """
    Generates all points on a path defined by x and y coordinates.

    Args:
        xys (numpy.ndarray): Path coordinates.

    Returns:
        numpy.ndarray: Array containing all points on the path.

    """
    points = np.array([xys[0]])
    for idx in range(xys.shape[0] - 1):
        a = xys[idx]
        b = xys[idx + 1]
        rr, cc = line(a[0], a[1], b[0], b[1])
        points = np.concatenate((points, (np.column_stack((rr, cc)))))
    new_array = np.array([tuple(row) for row in points])
    _, idx = np.unique(new_array, axis=0, return_index=True)
    return new_array[np.sort(idx)]


def medial_axis_path(
    mesh: np.ndarray, start: np.ndarray, end: np.ndarray, scale=1
) -> tuple[np.ndarray, float]:
    """
    converts image to graph module networkx, computes shortest path between two points where
    length and distance to the next zero points is considered and returns shortest path

    :param mesh: array of int(0,1) representing the thresholded image
    :param start: start point shape: (1x2)
    :param end:  end point shape: (1x2)
    :param scale: scaling factor units [scaling] = mum/pixel
    :return: shortest path shape (Nx2)
    """
    a = np.stack((start, end), axis=0)
    if (mesh[a[:, 0], a[:, 1]]).all():
        len_y = len(mesh[:, 0])
        len_x = len(mesh[0, :])
        G = nx.Graph()
        flatimg = mesh.flatten()
        side_length = len_x
        scale = scale
        pos = {}
        # create Graph for flat image
        for i, val in enumerate(flatimg):
            if val == 1:
                G.add_node(i)
                pos[i] = (i % side_length, len_x - mt.floor(i / side_length))

                if i < len(flatimg) - side_length:
                    if (i + 1) % side_length != 0 and flatimg[
                        i + 1
                    ] == 1:  # add horizontal edges
                        G.add_edge(i, i + 1, weight=scale * 1)

                    if (i + side_length) < side_length**2 and flatimg[
                        i + side_length
                    ] == 1:  # add vertical edges
                        G.add_edge(i, i + side_length, weight=scale * 1)

                    if (
                        (i + 1 + side_length) < side_length**2
                        and (i + 1 + side_length) % side_length != 0
                        and flatimg[i + side_length + 1] == 1
                    ):  # one diagonal direction
                        G.add_edge(i, i + 1 + side_length, weight=scale * mt.sqrt(2))

                    if (
                        (i - 1 + side_length) < side_length**2
                        and (i - 1 + side_length) % side_length != side_length - 1
                        and flatimg[i - 1 + side_length] == 1
                    ):  # the other diagonal direction
                        G.add_edge(i, i - 1 + side_length, weight=scale * mt.sqrt(2))

        """
        find shortest path between two nodes and draw network with path in different color
        """

        node1, node2 = start[0] * len_x + start[1], end[0] * len_x + end[1]
        path = nx.dijkstra_path(
            G, node1, node2, weight="weight"
        )  # shortest path as list of nodes
        w = nx.path_weight(G, path, weight="weight")

        """
        for each node find distance to dendrite boundary
        """
        G2 = G.copy()
        border_dist = 0
        while (
            len(G2.nodes) > 0
        ):  # iteratively remove outermost layer of graph until nothing is left
            neighbor_numbers = np.array([len(list(G2.neighbors(n))) for n in G2.nodes])
            border_indices = np.where(neighbor_numbers < 8)[
                0
            ]  # less than 8 neighbors means the node is on the outside boundary
            border_nodes = np.array(list(G2.nodes))[border_indices]
            for n in border_nodes:
                G.nodes[n][
                    "border_dist"
                ] = border_dist  # mark the distance from boundary in original graph
            G2.remove_nodes_from(border_nodes)
            border_dist += 1

        """
        find shortest path according to adjusted weights (middle of dendrite gets rewarded)
        """
        max_weight = 5
        middle_factor = 0.1
        for u, v in G.edges:
            w1, w2 = G.nodes[u]["border_dist"], G.nodes[v]["border_dist"]
            G.edges[(u, v)]["adj_weight"] = (
                np.abs((max_weight - (w1 + w2) * middle_factor))
                * G.edges[(u, v)]["weight"]
            )  # edges between nodes far away from boundary have lower weight

        path = nx.dijkstra_path(
            G, node1, node2, weight="adj_weight"
        )  # find shortest path according to new weights

        # from graph representation -> image indices

        coords = np.zeros((len(path), 2))  # dummy array to save arrays in
        for ind, val in enumerate(path):
            y = path[ind] // len_x
            x = path[ind] - len_x * y
            coords[ind] = x, y

        coords[:, 0] = gaussian_filter1d(coords[:, 0], mode="nearest", sigma=10)
        coords[:, 1] = gaussian_filter1d(coords[:, 1], mode="nearest", sigma=10)

        # calculate length of path in pixels
        length = GetLength(coords)
        return coords, length * scale
    else:
        print("points were not on the dendrit, press Go again!")

def GetLength(coords):
    length = 0
    for i in range(1,len(coords)):
        diff = np.linalg.norm(coords[i]-coords[i-1])
        length += diff

    return length

def downsampling_max_pool(img: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
    """
    function is downsampling with a max pooling from pytorch
    and converts back to numpy array
    :param img: thresholded image
    :param kernel_size: kernel size if the max pooling
    :param stride: shifting steps of the kernel
    :return: downsampled image
    """
    tensor = torch.from_numpy(img)
    tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.float()

    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    result = pool(tensor)
    np_arr = result.cpu().detach().numpy()
    np_arr = np_arr[0, :, :]
    return np_arr
