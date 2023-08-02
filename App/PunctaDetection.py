#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:22:20 2021

@author: surbhitwagle
"""

import numpy as np
import os
from math import sqrt
import matplotlib.pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh

from skimage.draw import polygon, disk

from .PathFinding import GetAllpointsonPath
import json
import csv

class Puncta:

    def __init__(self,location,radius,stats,between_cp,distance,struct,channel,snapshot):
        self.location  = location
        self.radius    = radius
        self.max       = stats[0]
        self.min       = stats[1]
        self.mean      = stats[2]
        self.std       = stats[3]
        self.median    = stats[4]
        self.between   = between_cp
        self.distance  = distance
        self.struct    = struct
        self.channel   = channel
        self.snapshot  = snapshot

class PunctaDetection:
    """
    class that holds meta data for puncta detection and methods for puncta stats calculations
    """

    def __init__(self, SimVars, tiff_Arr, somas, dendrites, dend_thresh=0.75,soma_thresh=0.5):
        self.Dir = SimVars.Dir
        self.tiff_Arr = tiff_Arr
        self.somas = somas  
        self.dendrites = dendrites  
        self.channels = SimVars.Channels
        self.snaps    = SimVars.Snapshots
        self.scale = SimVars.Unit  
        self.dend_thresh = dend_thresh
        self.soma_thresh = soma_thresh
        self.SimVars = SimVars

    def isBetween(self, a, b, c):
        """
        function that checks if c lies on perpendicular space between line segment a to b
        input: roi consecutive points a,b and puncta center c
        output: True/False
        """
        sides = np.zeros(3)
        sides[0] = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2  # ab
        original = sides[0]
        sides[1] = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2  # bc
        sides[2] = (c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2  # ca
        sides = np.sort(sides)
        if sides[2] > (sides[1] + sides[0]) and sides[2] != original:
            return False

        return True

    def Perpendicular_Distance_and_POI(self, a, b, c):
        """
        distance between two parallel lines, one passing (line1, A1 x + B1 y + C1 = 0) from a and b
        and second one (line 2, A1 x + B1 y + C2 = 0) parallel to line1 passing from c is given
        |C1-C2|/sqrt(A1^2 + B1^2)

        input: roi consecutive points a,b and puncta center c
        output: Perpendicular from line segment a to b and point of intersection at the segment
        """
        m = (a[1] - b[1]) / (a[0] - b[0] + 1e-18)
        if m == 0:
            m = 1e-9
        c1 = a[1] - m * a[0]
        c2 = c[1] - m * c[0]
        dist = np.absolute(c1 - c2) / np.sqrt(1 + m**2)
        m_per = -1 / m
        c3 = c[1] - m_per * c[0]
        x_int = (c3 - c1) / (m - m_per) * 1.0
        y_int = (m_per * x_int + c3) * 1.0

        ax_int = np.sqrt((a[0] - x_int) ** 2 + (a[1] - y_int))
        bx_int = np.sqrt((b[0] - x_int) ** 2 + (b[1] - y_int))
        ab = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        return x_int, y_int, dist

    def GetClosestRoiPoint(self, dendrite, point):
        """
        function that finds closest roi point if point is not on dendrite
        input: dendrite rois,point
        output: distance from the origin of the dendrite
        """
        min_dist = 10**18
        prev = [dendrite[0][0], dendrite[1][0]]
        dist_from_origin = 0
        closest_p = [0, 0]
        closed_p_idx = 0
        for idx, x in enumerate(dendrite[0][:]):
            y = dendrite[1][idx]
            a = [x, y]
            dist = np.sqrt((point[1] - a[1]) ** 2 + (point[0] - a[0]) ** 2)
            if dist < min_dist:
                min_dist = dist
                dist_from_origin += np.sqrt(
                    (prev[1] - a[1]) ** 2 + (prev[0] - a[0]) ** 2
                )
                closest_p = a
                closed_p_idx = idx
            prev = a
        return dist_from_origin

    def Is_On_Dendrite(self, dendrite, point, max_dist):
        """
            function that checks on which segment of the dendrite the point is present (if)
            input: dendrite,point,max_dist
            output: True/False and scaled distance from the origin of the dendrite
        """
        length_from_origin = 0
        prev_distance = 10**20
        for idx, x in enumerate(dendrite[0][:-1]):
            y = dendrite[1][idx]
            a = [x, y]
            b = [dendrite[0][idx + 1], dendrite[1][idx + 1]]
            if self.isBetween(a, b, point):
                x_int, y_int, distance = self.Perpendicular_Distance_and_POI(
                    a, b, point
                )
                if distance <= max_dist:
                    length_from_origin += np.sqrt(
                        (y_int - a[1]) ** 2 + (x_int - a[0]) ** 2
                    )
                    return True, length_from_origin * self.scale
            length_from_origin += np.sqrt((b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2)

        length_from_origin = self.GetClosestRoiPoint(dendrite, point)
        return False, length_from_origin * self.scale

    # set somatic = False for dendritic punctas
    def GetPunctaStats(self, x, y, r, original_img):
        """
        function that claculates the stats of gaussian puncta centered at x,y with radius r
        input: x,y, r and original image called by PunctaDetection class object
        output: list that includes the max, min,mean,std and median of the pixels in circle at x,y with radius r
        """
        #
        img = np.zeros(original_img.shape, dtype=np.uint8)
        rr, cc = disk((y, x), r, shape=original_img.shape)
        img[rr, cc] = 1
        f_img = np.multiply(original_img, img)
        f_img_data = original_img[np.nonzero(f_img)]
        puncta_stats = [
            f_img_data.max(),
            f_img_data.min(),
            f_img_data.mean(),
            f_img_data.std(),
            np.median(f_img_data),
        ]
        return puncta_stats

    def GetPunctas(self,Soma=True):
        """
            function that does the puncta detection
            input: none, called by PunctaDetection class object
            output: two dictionaries that stores list of puncta stats for each puncta element wise (soma/dendrite)
        """
        NoDendrite = False
        all_c_t_somatic_puncta = []
        all_c_t_dendritic_puncta = []
        for t in range(self.snaps):
            all_c_somatic_puncta = []
            all_c_dendritic_puncta = []
            for ch in range(self.channels):

                orig_img = self.tiff_Arr[t, ch, :, :].astype(float)
                if(Soma):
                    somatic_puncta,anti_soma   = self.GetPunctasSoma(orig_img,ch,t)
                    all_c_somatic_puncta.append(somatic_puncta)
                else:
                    anti_soma = np.ones(np.shape(orig_img), "uint8")
                try:
                    dendritic_puncta = self.GetPunctasDend(orig_img,anti_soma,ch,t)
                except:
                    NoDendrite = True
                    dendritic_puncta = []

                all_c_dendritic_puncta.append(dendritic_puncta)
            all_c_t_somatic_puncta.append(all_c_somatic_puncta)
            all_c_t_dendritic_puncta.append(all_c_dendritic_puncta)
        if(not NoDendrite):
            self.SimVars.frame.set_status_message.setText("Punctas are available on all snaphshots/channels")
        else:
            self.SimVars.frame.set_status_message.setText("Punctas are available on all snaphshots/channels, but there was no dendrite, so no dendritic puncta")
        return all_c_t_somatic_puncta, all_c_t_dendritic_puncta

    def GetPunctasSoma(self,orig_img,ch,t_snape):
        """Detects and returns somatic puncta in the given image.

        Performs puncta detection on the soma regions of the image and returns the detected puncta.

        Args:
            orig_img: The original image in which puncta are to be detected.

        Returns:
            somatic_puncta: A list of Puncta objects representing the detected somatic puncta.
            anti_soma: An anti-soma image obtained by subtracting soma regions from the original image.
        """
        somatic_puncta = []

        soma_img = np.zeros(np.shape(orig_img), "uint8")
        anti_soma = np.ones(np.shape(orig_img), "uint8")

        for i,soma_instance in enumerate(self.somas):
            lsm_img = np.zeros(np.shape(orig_img), "uint8")

            xs = soma_instance[:, 0]
            ys = soma_instance[:, 1]

            rr, cc = polygon(ys, xs, lsm_img.shape)
            lsm_img[rr, cc] = 1

            anti_soma = np.multiply(anti_soma, 1 - lsm_img)
            soma_img = np.multiply(orig_img, lsm_img)
            t = np.max(orig_img[rr,cc])*self.soma_thresh
            blobs_log = blob_log(soma_img, threshold=t,max_sigma=1)
            blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

            for blob in blobs_log:
                y, x, r = blob
                puncta_stats = self.GetPunctaStats(x, y, r, orig_img)
                sp = Puncta([x,y],r,puncta_stats,False,0,i,ch,t_snape)
                somatic_puncta.append(sp)

        return somatic_puncta,anti_soma

    def GetPunctasDend(self,orig_img,anti_soma,ch,t_snape):

        """Detects and returns dendritic puncta in the given image.

        Performs puncta detection on the dendrite regions of the image and returns the detected puncta.

        Args:
            orig_img: The original image in which puncta are to be detected.
            anti_soma: The anti-soma image obtained by subtracting soma regions from the original image.

        Returns:
            dendritic_puncta: A list of Puncta objects representing the detected dendritic puncta.
        """

        dendritic_puncta = []
        lsm_img = np.zeros(np.shape(orig_img), "uint8")

        dendrite_img = np.zeros(np.shape(orig_img), "uint8")
        dilated = np.zeros(np.shape(orig_img), "uint8")

        for i,dendrite_instance in enumerate(self.dendrites):
            dilated = dendrite_instance.get_dendritic_surface_matrix()
            dilated = np.multiply(anti_soma, dilated)
            xy = GetAllpointsonPath(dendrite_instance.control_points)[:, :]
            xs = xy[:, 0]
            ys = xy[:, 1]
            ## uncomment if you don't want to repeat dendritic punctas in overlapping dendritic parts
            # anti_soma = np.multiply(anti_soma,1 - dilated)
            dend_img = np.multiply(dilated, orig_img)
            filtered_dend_img = dend_img[np.nonzero(dend_img)]
            t = np.quantile(filtered_dend_img, self.dend_thresh)
            dend_blobs_log = blob_log(dend_img, threshold=t,max_sigma=1)
            dend_blobs_log[:, 2] = dend_blobs_log[:, 2] * sqrt(2)
            dp = []
            for blob in dend_blobs_log:
                y, x, r = blob
                on_dendrite, distance_from_origin = self.Is_On_Dendrite(
                    [xs, ys], [x, y], dendrite_instance.dend_stat[:,2].max()
                )
                puncta_stats = self.GetPunctaStats(x, y, r, orig_img)
                dp = Puncta([x,y],r,puncta_stats,on_dendrite,distance_from_origin,i,ch,t_snape)
                dendritic_puncta.append(dp)

        return dendritic_puncta

def save_puncta(puncta_Dir,punctas,xLims):
    """Saves the detected puncta to files.

    This method creates a directory for puncta files and subdirectories for different parameters.
    It retrieves the current slider values for half width, dendritic threshold, and somatic threshold.
    The somatic and dendritic punctas are obtained from the punctas list and flattened.
    The somatic punctas are saved to a JSON file under the 'soma_puncta.json' filename.
    The dendritic punctas are saved to a JSON file under the 'dend_puncta.json' filename.
    Both files are stored in the corresponding subdirectory of the puncta directory.
    """
    
    if(len(xLims[0])==0):
        Lims = np.array(0)
    else:
        Lims = np.array([xLims[0][0],xLims[1][0]])

    somatic_punctas,dendritic_punctas = punctas[0],punctas[1]
    somatic_punctas_flat = [item for sublist in somatic_punctas for subsublist in sublist for item in (subsublist if isinstance(subsublist, list) else [subsublist])]
    dendritic_punctas_flat =  [item for sublist in dendritic_punctas for subsublist in sublist for item in (subsublist if isinstance(subsublist, list) else [subsublist])]
    try:
        for sp in somatic_punctas_flat:
            sp.location = (location - Lims).tolist()
    except:
        pass
    try:
        for dp in dendritic_punctas_flat:
            dp.location = (location - Lims).tolist()
    except:
        pass

    with open(
        puncta_Dir + "soma_puncta.json",
        "w",
    ) as f:
        json.dump([vars(P) for P in somatic_punctas_flat], f, indent=4)
    with open(
        puncta_Dir + "dend_puncta.json",
        "w",
    ) as f:
        json.dump([vars(P) for P in dendritic_punctas_flat], f, indent=4)

    PunctaSave_csv(puncta_Dir,somatic_punctas_flat,dendritic_punctas_flat)

def PunctaSave_csv(Dir,somatic_punctas_flat,dendritic_punctas_flat):
    """
    Saves somatic and dendritic puncta data to separate CSV files.

    Args:
        Dir (str): Directory path where the CSV files will be saved.
        somatic_punctas_flat (list): List of somatic puncta objects.
        dendritic_punctas_flat (list): List of dendritic puncta objects.

    Returns:
        None
    """
    custom_header = ['','channel','snapshot','location','radius','max','min','mean','std','median','distance']

    csv_file_path = Dir+'soma_puncta.csv'
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(custom_header) 
        for i,p in enumerate(somatic_punctas_flat):
            row = ['Puncta: '+str(i),p.channel,p.snapshot,str(p.location),
                   p.radius,p.max,p.min,p.mean,p.std,p.median,p.distance]
            writer.writerow(row)

    csv_file_path = Dir+'dend_puncta.csv'
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(custom_header) 
        for i,p in enumerate(dendritic_punctas_flat):
            row = ['Puncta: '+str(i),p.channel,p.snapshot,str(p.location),
                   p.radius,p.max,p.min,p.mean,p.std,p.median,p.distance]
            writer.writerow(row)
