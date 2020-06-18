#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: plot_tropomi.py
WORKS FOR PLOTTING OVER TORONTO USING A KML FILE
"""

# Preamble
from simplekml import (Kml, OverlayXY, ScreenXY, Units, RotationXY,
                       AltitudeMode, Camera)
import open_tropomi as ot
import grid_tropomi as gt
import points_of_interest as poi
from paths import *
import os
import glob
import pickle
import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
import time
import datetime
import simplekml
from datetime import timedelta
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.colorbar import colorbar


def make_kml(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
             figs, colorbar=None, **kw):
    """TODO: LatLon bbox, list of figs, optional colorbar figure,
    and several simplekml kw...
    """
    
    # Initiatlize KML file and remove altitude, roll, tilt, and altitudemode
    # if keys do not exist, give default values
    kml = Kml()
    
    altitude = kw.pop('altitude', 2e7)
    roll = kw.pop('roll', 0)
    tilt = kw.pop('tilt', 0)
    altitudemode = kw.pop('altitudemode', AltitudeMode.relativetoground)
    
    # Create virtual camera that views the scene
    camera = Camera(latitude=np.mean([urcrnrlat, llcrnrlat]),
                    longitude=np.mean([urcrnrlon, llcrnrlon]),
                    altitude=altitude, roll=roll, tilt=tilt,
                    altitudemode=altitudemode)
    kml.document.camera = camera
    
    draworder = 0
    for fig in figs:  # NOTE: Overlays are limited to the same bbox.
        draworder += 1
        ground = kml.newgroundoverlay(name='GroundOverlay')
        ground.draworder = draworder
        ground.visibility = kw.pop('visibility', 1)
        ground.name = kw.pop('name', 'overlay')
        ground.color = kw.pop('color', '9effffff')
        ground.atomauthor = kw.pop('author', 'ocefpaf')
        ground.latlonbox.rotation = kw.pop('rotation', 0)
        ground.description = kw.pop('description', 'Matplotlib figure')
        ground.gxaltitudemode = kw.pop('gxaltitudemode',
                                       'clampToSeaFloor')
        ground.icon.href = fig
        ground.latlonbox.east = llcrnrlon
        ground.latlonbox.south = llcrnrlat
        ground.latlonbox.north = urcrnrlat
        ground.latlonbox.west = urcrnrlon

    if colorbar:  # Options for colorbar are hard-coded (to avoid a big mess).
        screen = kml.newscreenoverlay(name='ScreenOverlay')
        screen.icon.href = colorbar
        screen.overlayxy = OverlayXY(x=0, y=0,
                                     xunits=Units.fraction,
                                     yunits=Units.fraction)
        screen.screenxy = ScreenXY(x=0.015, y=0.075,
                                   xunits=Units.fraction,
                                   yunits=Units.fraction)
        screen.rotationXY = RotationXY(x=0.5, y=0.5,
                                       xunits=Units.fraction,
                                       yunits=Units.fraction)
        screen.size.x = 0
        screen.size.y = 0
        screen.size.xunits = Units.fraction
        screen.size.yunits = Units.fraction
        screen.visibility = 1

    kmzfile = kw.pop('kmzfile', 'overlay.kmz')
    kml.savekmz(kmzfile)


def gearth_fig(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, pixels=1024):
    """Return a Matplotlib `fig` and `ax` handles for a Google-Earth Image."""
    aspect = np.cos(np.mean([llcrnrlat, urcrnrlat]) * np.pi/180.0)
    xsize = np.ptp([urcrnrlon, llcrnrlon]) * aspect
    ysize = np.ptp([urcrnrlat, llcrnrlat])
    aspect = ysize / xsize

    if aspect > 1.0:
        figsize = (10.0 / aspect, 10.0)
    else:
        figsize = (10.0, 10.0 * aspect)

    if False:
        plt.ioff()  # Make `True` to prevent the KML components from poping-up.
    fig = plt.figure(figsize=figsize,
                     frameon=False,
                     dpi=pixels//10)
    # KML friendly image.  If using basemap try: `fix_aspect=False`.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(llcrnrlon, urcrnrlon)
    ax.set_ylim(llcrnrlat, urcrnrlat)
    return fig, ax


pixels = 1024 * 10

fig, ax = gearth_fig(llcrnrlon=-80.347,
                     llcrnrlat=42.65107,
                     urcrnrlon=-78.347,
                     urcrnrlat=44.65107,
                     pixels=pixels)

input_files = os.path.join(tropomi_pkl_month, '{}/2020*'.format('toronto'))
for test_file in sorted(glob.glob(input_files)):
    infile = open(test_file, 'rb')
    ds = pickle.load(infile)
    infile.close()

cs = ds.isel(time=0).plot.pcolormesh(x='longitude', y='latitude',
                                     robust=True,
                                     add_colorbar=False)
ax.set_axis_off()
fig.savefig('overlay1.png', transparent=False, format='png')


fig = plt.figure(figsize=(2.0, 4.0), facecolor=None, frameon=False)
ax = fig.add_axes([0.0, 0.05, 0.2, 0.9])
cb = fig.colorbar(cs, cax=ax)
cb.set_label(r'NO$_2$ Tropospheric Vertical Column',
             rotation=-90, color='k', labelpad=20)
# Change transparent to True if your colorbar is not on space :)
fig.savefig('legend.png', transparent=False, format='png')


make_kml(llcrnrlon=-80.347,
         llcrnrlat=42.65107,
         urcrnrlon=-78.347,
         urcrnrlat=44.65107,
         figs=['overlay1.png'], colorbar='legend.png',
         kmzfile='tor_2020.kmz', name='Mean NO2 TVCD')