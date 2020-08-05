import xarray as xr
import numpy as np
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt

from paths import *
import points_of_interest as poi

# unpickle files


def create_colorbar(im, ax, label, orientation='horizontal',
                    fmt_num=-5, mathbool=True, resid=False):
    """
    Return a colorbar to plot for TROPOMI NO2 TVCD.

    Args:
        im: the mathplotlib.cm.ScalarMappable described by this colorbar.
        ax (Axes): list of axes.
        label (str): label for colorbar.
        orientation (str): vertical or horizontal.
        fmt_num (num): exponent for scientific notation.
        mathbool (bool): toggle mathText.
        resid (bool): Toggle residual difference.

    Returns:
        cbar (colorbar)
    """

    class OOMFormatter(matplotlib.ticker.ScalarFormatter):
        def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
            self.oom = order
            self.fformat = fformat
            matplotlib.ticker.ScalarFormatter.__init__(
                self, useOffset=offset, useMathText=mathText)

        def _set_order_of_magnitude(self):
            self.orderOfMagnitude = self.oom

        def _set_format(self, vmin=None, vmax=None):
            self.format = self.fformat
            if self._useMathText:
                self.format = r'$\mathdefault{%s}$' % self.format

    if resid:
        asp = 20
    else:
        asp = 45

    cbar = plt.colorbar(mappable=im, ax=ax, orientation=orientation,
                        format=OOMFormatter(fmt_num, mathText=mathbool),
                        aspect=asp)
    cbar.ax.xaxis.get_offset_text().set_visible(False)
    cbar.set_label(
        r"%s ($10^{%s}$ mol/m$^2$)" % (label, fmt_num), labelpad=8, fontsize=10)

    return cbar


def plot_single(time, data_type, wind_type, city='toronto'):

    f = '{}/gridded/{}_{}_{}_70'.format(
        city, poi.nomen_dict[time], poi.nomen_dict[data_type], wind_type)
    if data_type == 'rotated':
        f = rotated_pkl + f
    elif data_type == 'cartesian':
        f = cartesian_pkl + f

    with open(f, 'rb') as infile:
        ds = pickle.load(infile)
    if wind_type in list(ds.keys()):    
        ds = ds[wind_type]

        fig, ax = plt.subplots(1, 1)
        im = ax.contourf(ds.x.values, ds.y.values, ds.no2_avg, cmap='jet')
        cb = create_colorbar(im=im, ax=ax, label='Mean NO$_2$ TVCD', orientation='vertical')
        cb.ax.yaxis.get_offset_text().set_visible(False)
        ax.set_xlabel(r'$\longleftarrow$ upwind      x (km)      downwind $\longrightarrow$')
        ax.set_ylabel(r'y (km) $\longrightarrow$')
        ax.scatter(0, 0, marker='*', color='black')
        
        d1 = ds.attrs['timeframe'][:10]
        d2 = ds.attrs['timeframe'][20:30]
        ws1 = poi.wind_type[wind_type][0]
        ws2 = poi.wind_type[wind_type][1]
        
        plt.title(r'Average NO$_2$ TVCD over Toronto ({}-{}m/s)'.format(ws1, ws2) + '\n {} to {}'.format(d1, d2))
        plt.show()

        return ds
    
    else: 
        print('no data')


def plot_multiple(time, data_type='rotated', city='toronto', dist=70):

    wind_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # colourbar formatting
    class OOMFormatter(matplotlib.ticker.ScalarFormatter):
        def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
            self.oom = order
            self.fformat = fformat
            matplotlib.ticker.ScalarFormatter.__init__(
                self, useOffset=offset, useMathText=mathText)

        def _set_order_of_magnitude(self):
            self.orderOfMagnitude = self.oom

        def _set_format(self, vmin=None, vmax=None):
            self.format = self.fformat
            if self._useMathText:
                self.format = r'$\mathdefault{%s}$' % self.format

    # create figure
    figure = plt.figure(figsize=(12, 6.5))
    grid = ImageGrid(figure, 111,          # as in plt.subplot(111)
                     nrows_ncols=(2, 4),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="3%",
                     cbar_pad=0.1)

    # iterate over each file
    for i, (wt, ax) in enumerate(zip(wind_types, grid)):
        f = '{}/gridded/{}_{}_{}_{}'.format(
            city, poi.nomen_dict[time], poi.nomen_dict[data_type], wt, dist)
        if data_type == 'rotated':
            f = rotated_pkl + f
        elif data_type == 'cartesian':
            f = cartesian_pkl + f

        with open(f, 'rb') as infile:
            ds = pickle.load(infile)
            ds = ds[wt]
            
        levels = np.linspace(0.0, 12e-5, 12)
        im = ax.contourf(ds.x.values, ds.y.values, ds.no2_avg, levels=levels, cmap='jet')
        ax.set_aspect('equal')
        ax.scatter(0, 0, marker='*', color='black')  # mark toronto

        # label subplot with wind speeds
        ax.annotate('{}-{}m/s'.format(poi.wind_type[wt][0], poi.wind_type[wt][1]),
                    xy=(0.05, 0.9), xycoords='axes fraction', color='white')

        if i == 0 or i == 4:
            ax.set_ylabel('y (km)')
        if i==4 or i == 5 or i == 6 or i == 7 :
            ax.set_xlabel(
                r'$\leftarrow$ upwind   x (km)  downwind $\rightarrow$')

    # create colourbar
    cbar = ax.cax.colorbar(im, format=OOMFormatter(-5, mathText=True))
    cbar.set_label_text(r'Mean NO$_2$ TVCD ($10^{-5}$ mol/m$^2$)')
    ax.cax.toggle_label(True)
    
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout(rect=[0, 0, 0.90, 0.98])

    # create title
    d1 = ds.attrs['timeframe'][:10]
    d2 = ds.attrs['timeframe'][20:30]
    figure.suptitle(r'Average NO$_2$ TVCD over Toronto' + '\n {} to {}'.format(d1, d2),
                    fontsize=14)
    plt.show()

    # Ask to save file
    if __name__ == '__main__':
        is_save = str(
            input('Do you want to save a png of this figure? \n (Y/N)'))
        if is_save == 'Y' or is_save == 'y':
            city_abbrs = {'toronto': 'TOR',
                          'montreal': 'MTL',
                          'vancouver': 'VAN',
                          'los_angeles': 'LAL',
                          'new_york': 'NYC'}
            nomen_dict = poi.nomen_dict

            print('Saving png for {} for {} to {}'.format(city, d1, d2))
            pngfile = '{0}.png'.format('../figures/{city}/{time}_{data_type}_{wind_type}_{dist}_{date}'.format(city=city, time=nomen_dict[time],
                                                               data_type=nomen_dict[data_type],
                                                               wind_type='ALL',
                                                               dist=str(dist), date=dt.datetime.now().strftime('_%Y%m%dT%H%M%S')))
            figure.savefig(pngfile, dpi=300)



if __name__ == '__main__':
    ds = plot_single(time='may_20', data_type='rotated', wind_type='C')
    # print(ds)
    # time='may_20'
    # data_type='rotated'
    # wt='A'
    # dist=70
    # city='toronto'
    # f = '{}/gridded/{}_{}_{}_{}'.format(
    #     city, poi.nomen_dict[time], poi.nomen_dict[data_type], wt, dist)
    # if data_type == 'rotated':
    #     f = rotated_pkl + f
    # elif data_type == 'cartesian':
    #     f = cartesian_pkl + f

    # with open(f, 'rb') as infile:
    #     ds = pickle.load(infile)
    #     ds = ds[wt]