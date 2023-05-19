#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:22:20 2021

@author: surbhitwagle
"""

import numpy as np
import os
from math import sqrt
import tifffile as tf
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh

from os.path import exists
from skimage.draw import line, polygon, ellipse, disk
from skimage import morphology

class Puncta:

    def __init__(self,location,radius,stats,between_cp,distance,struct):
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
        self.channel   = 0
        self.snapshot  = 0

class PunctaDetection:
    """
    class that holds meta data for puncta detection and methods for puncta stats calculations
    """

    def __init__(self, SimVars, tiff_Arr, somas, dendrites, channels, width=5,dend_thresh=0.75,soma_thresh=0.5):
        self.Dir = SimVars.Dir
        self.tiff_Arr = tiff_Arr
        self.somas = (
            somas  # should be a dict with name of dendrite as key and polygone as value
        )
        self.dendrites = dendrites  # should be a dict with name of dendrite as key and ROIs as values
        self.channels = SimVars.Channels
        self.snaps    = SimVars.Snapshots
        self.scale = SimVars.Unit  # micons/pixel
        self.width = width / self.scale
        self.dend_thresh = dend_thresh
        self.soma_thresh = soma_thresh

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

    def Is_On_Dendrite(self, dendrite_name, dendrite, point, max_dist):
        """
            function that checks on which segment of the dendrite the point is present (if)
            input: dendrite_name,dendrite,point,max_dist
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
                    return True, length_from_origin / self.scale
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
        all_c_t_somatic_puncta = []
        all_c_t_dendritic_puncta = []
        for t in range(self.snaps):
            all_c_somatic_puncta = []
            all_c_dendritic_puncta = []
            for ch in range(self.channels):

                orig_img = self.tiff_Arr[t, ch, :, :].astype(float)
                if(Soma):
                    somatic_puncta,anti_soma   = self.GetPunctasSoma(orig_img)
                    all_c_somatic_puncta.append(somatic_puncta)
                else:
                    anti_soma = np.ones(np.shape(orig_img), "uint8")
                dendritic_puncta = self.GetPunctasDend(orig_img,anti_soma)
                all_c_dendritic_puncta.append(dendritic_puncta)
            for dp in all_c_dendritic_puncta:
                for d in dp:
                    d.snapshot = t
                    d.channel  = ch
            for sp in all_c_somatic_puncta:
                for s in sp:
                    s.snapshot = t
                    s.channel  = ch
            all_c_t_somatic_puncta.append(all_c_somatic_puncta)
            all_c_t_dendritic_puncta.append(all_c_dendritic_puncta)
        return all_c_t_somatic_puncta, all_c_t_dendritic_puncta

    def GetPunctasSoma(self,orig_img):
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

        for i,soma in enumerate(self.somas.keys()):
            lsm_img = np.zeros(np.shape(orig_img), "uint8")
            soma_instance = self.somas[soma]

            xs = soma_instance[:, 0]
            ys = soma_instance[:, 1]

            rr, cc = polygon(ys, xs, lsm_img.shape)
            lsm_img[rr, cc] = 1

            anti_soma = np.multiply(anti_soma, 1 - lsm_img)
            soma_img = np.multiply(orig_img, lsm_img)

            t = np.quantile(orig_img[rr, cc], self.soma_thresh)
            
            blobs_log = blob_log(soma_img, threshold=t)
            blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

            for blob in blobs_log:
                y, x, r = blob
                puncta_stats = self.GetPunctaStats(x, y, r, orig_img)
                sp = Puncta([x,y],r,puncta_stats,False,0,i)
                somatic_puncta.append(sp)

        return somatic_puncta,anti_soma

    def GetPunctasDend(self,orig_img,anti_soma):

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

        for i,dendrite in enumerate(self.dendrites.keys()):

            dendrite_instance = self.dendrites[dendrite]
            xs = dendrite_instance[:, 0]
            ys = dendrite_instance[:, 1]
            for lk in range(0, len(xs) - 1):
                rr, cc = line(
                    int(ys[lk]), int(xs[lk]), int(ys[lk + 1]), int(xs[lk + 1])
                )
                dendrite_img[rr, cc] = 1
            dilated = morphology.dilation(
                dendrite_img, morphology.disk(radius=self.width)
            )
            dilated = np.multiply(anti_soma, dilated)
            ## uncomment if you don't want to repeat dendritic punctas in overlapping dendritic parts
            # anti_soma = np.multiply(anti_soma,1 - dilated)
            dend_img = np.multiply(dilated, orig_img)
            filtered_dend_img = dend_img[np.nonzero(dend_img)]
            t = np.quantile(filtered_dend_img, self.dend_thresh)
            dend_blobs_log = blob_log(dend_img, threshold=t)
            dend_blobs_log[:, 2] = dend_blobs_log[:, 2] * sqrt(2)
            dp = []
            for blob in dend_blobs_log:
                y, x, r = blob
                on_dendrite, distance_from_origin = self.Is_On_Dendrite(
                    dendrite, [xs, ys], [x, y], self.width
                )
                puncta_stats = self.GetPunctaStats(x, y, r, orig_img)
                dp = Puncta([x,y],r,puncta_stats,on_dendrite,distance_from_origin,i)
                dendritic_puncta.append(dp)

        return dendritic_puncta
