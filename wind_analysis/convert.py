#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert.py

Script containing functions:
    - haversine(lat1, lon1, lat2, lon2, bearing=False)
    - convert(lat1, lon1, lat2, lon2, out_units='km')
    - rotate(pivot, point, angle)

Functions to convert lat/lon coordinates to xy-coordinates using the 
haversine formula given in https://www.movable-type.co.uk/scripts/latlong.html 
and rotate points on a Cartesian grid. 
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

################################

def haversine(point1, point2, bearing=False):
    """
    Use the haversine formula to calculate distance between two sets of 
    lat/lon points. 

    Reference: https://www.movable-type.co.uk/scripts/latlong.html

    Args:
        point1 (float, float): latitude and longitude of first point.
        point2 (float, float): latitude and longitude of second point.
        bearing (bool): if True, bearing to each final point will be returned.

    Returns:
        d (float): distance (in km) between initial and final lat/lon.
        bear (float): bearing to final point if bearing is True.        
    """
    # convert to radians
    lat1, lon1 = point1
    lat2, lon2 = point2 
    
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)

    # difference in lat/lon between the two locations
    dlat_r = np.radians(lat2 - lat1)
    dlon_r = np.radians(lon2 - lon1)

    # calculate distance
    a1 = np.power(np.sin(dlat_r / 2.), 2.)
    a2 = np.cos(lat1_r) * np.cos(lat2_r) * np.power(np.sin(dlon_r / 2.), 2.)
    a = a1 + a2
    c = 2. * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = abs(6371. * c)
    # calculate bearing
    if bearing:
        y = np.sin(dlon_r) * np.cos(lat2_r)
        x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
        bear = np.degrees(np.arctan2(x, y))

        if bear < 0:
            bear += 360.

        return (d, bear)
    else:
        return d

################################

def convert_to_cartesian(point1, point2, out_units='km'):
    """
    Convert lat/lon coordinates to Cartesian.

    Args:
        point1 (tuple of floats): latitude and longitude of first point.
        point2 (tuple of floats): latitude and longitude of second point.
        out_units (str): units for output. Default: 'km'
    Return
        x, y (floats): xy-coordinates of lat2/lon2 relative to lat1/lon1.
    """
    dist, bear = haversine(point1, point2, bearing=True)
    x = dist * np.cos(np.radians(bear))
    y = dist * np.sin(np.radians(bear))

    if out_units == 'km':
        return x, y
    elif out_units == 'm':
        return 1000 * x, 1000 * y
    else:
        raise ValueError('out_units must be one of the following: "km", "m"')

################################

def rotate(pivot, point, angle):
    """
    Return xy-coordinate for an initial point (x, y) rotated by angle (in
    degrees) around a pivot (x, y).

    Args:
        pivot (tuple of floats): pivot point of rotation.
        point (tuple of floats): xy-coordinates of point to be rotated.
        angle (float): angle to rotate the point around the pivot.
            Must be in degrees.
    Returns:
        (xnew, ynew): xy-coordinates of rotated point.
    """
    # to rotate cw, need negative bearing
    s = np.sin(np.radians(angle))
    c = np.cos(np.radians(angle))

    # translate point back to origin
    x = point[0] - pivot[0]
    y = point[1] - pivot[1]

    # rotate clockwise to the x-axis
    xnew = x * c + y * s
    ynew = x * (-s) + y * c

    # translate point back
    xnew += pivot[0]
    ynew += pivot[1]

    return (xnew, ynew)


if __name__ == '__main__':
    Point = namedtuple('Point', 'lon lat')
    cities = {'toronto': Point(-79.3832, 43.6532),
              'hamilton': Point(-79.8711, 43.2557),
              'montreal': Point(-73.5673, 45.5017),
              'new_york': Point(-74.0060, 40.7128),
              'vancouver': Point(-123.1207, 49.2827),
              'los_angeles': Point(-118.2437, 34.0522)}
    TORONTO = cities['toronto']
    HAMILTON = cities['hamilton']

    dist0, bear0 = haversine(
        (TORONTO[1], TORONTO[0]), (TORONTO[1], TORONTO[0]), bearing=True)
    dist1, bear1 = haversine(
        (TORONTO[1], TORONTO[0]), (HAMILTON[1], HAMILTON[0]), bearing=True)

    x0, y0 = dist0*np.cos(np.radians(bear0)), dist0*np.sin(np.radians(bear0))
    x1, y1 = dist1*np.cos(np.radians(bear1)), dist1*np.sin(np.radians(bear1))

    x = np.array([x0, x1])
    y = np.array([y0, y1])

    plt.scatter(x, y)
    plt.show()
