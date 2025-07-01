#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import numpy as np

# class monitor_linux:
#     import gi.repository
#     gi.require_version('Gdk', '3.0')
#     from gi.repository import Gdk
#
#     def __init__(self):
#         display = Gdk.Display.get_default()
#         screen = display.get_default_screen()
#         default_screen = screen.get_default()
#         num = default_screen.get_number()
#
#         self.h_mm = default_screen.get_monitor_height_mm(num)
#         self.w_mm = default_screen.get_monitor_width_mm(num)
#
#         self.h_pixels = default_screen.get_height()
#         self.w_pixels = default_screen.get_width()
#
#     def monitor_to_camera(self, x_pixel, y_pixel):
#
#         # assumes in-build laptop camera, located centered and 10 mm above display
#         # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
#         x_cam_mm = ((int(self.w_pixels/2) - x_pixel)/self.w_pixels) * self.w_mm
#         y_cam_mm = 10.0 + (y_pixel/self.h_pixels) * self.h_mm
#         z_cam_mm = 0.0
#
#         return x_cam_mm, y_cam_mm, z_cam_mm
#
#     def camera_to_monitor(self, x_cam_mm, y_cam_mm):
#         # assumes in-build laptop camera, located centered and 10 mm above display
#         # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
#         x_mon_pixel = np.ceil(int(self.w_pixels/2) - x_cam_mm * self.w_pixels / self.w_mm)
#         y_mon_pixel = np.ceil((y_cam_mm - 10.0) * self.h_pixels / self.h_mm)
#
#         return x_mon_pixel, y_mon_pixel

import wmi
from screeninfo import get_monitors
class monitor_windows:
    def __init__(self):
        # Get actual monitor physical size (in mm) from EDID
        self.w_mm, self.h_mm = self.get_monitor_size_mm()

        # Get monitor pixel resolution
        m = get_monitors()[0]  # primary monitor
        self.w_pixels = m.width
        self.h_pixels = m.height
#        self.w_pixels = 3840 Can hardcode if needed
#        self.h_pixels = 2160

    def get_monitor_size_mm(self):
#        return (609, 343) Can hardcode if needed
        w = wmi.WMI(namespace='root\\WMI')
        for m in w.WmiMonitorDescriptorMethods():
            raw = m.WmiGetMonitorRawEEdidV1Block(0)[0]
            edid_bytes = bytes(raw)
            width_cm = edid_bytes[21]
            height_cm = edid_bytes[22]
            return width_cm * 10, height_cm * 10  # mm
        raise RuntimeError("Could not read monitor EDID")

    def monitor_to_camera(self, x_pixel, y_pixel):
        # assumes in-build laptop camera, located centered and 10 mm above display
        x_cam_mm = ((int(self.w_pixels / 2) - x_pixel) / self.w_pixels) * self.w_mm
        y_cam_mm = 10.0 + (y_pixel / self.h_pixels) * self.h_mm
        z_cam_mm = 0.0
        return x_cam_mm, y_cam_mm, z_cam_mm

    def camera_to_monitor(self, x_cam_mm, y_cam_mm):
        x_mon_pixel = np.ceil(int(self.w_pixels / 2) - x_cam_mm * self.w_pixels / self.w_mm)
        y_mon_pixel = np.ceil((y_cam_mm - 10.0) * self.h_pixels / self.h_mm)
        return x_mon_pixel, y_mon_pixel
