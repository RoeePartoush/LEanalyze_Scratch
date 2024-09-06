from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy import stats
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
# import matplotlib import transforms
import matplotlib.pyplot as plt
import matplotlib.axes as pltax
import matplotlib.transforms as mtransforms
from scipy import ndimage
import scipy.interpolate as spi
from scipy.ndimage import gaussian_filter, median_filter, maximum_filter
from scipy.signal import medfilt

import pandas as pd

from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.io import fits
from astropy import wcs
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import Angle, SphericalRepresentation, CartesianRepresentation, UnitSphericalRepresentation
from astropy import wcs
from astropy.visualization import ZScaleInterval

import LeTools_Module as LeT
import LEtoolbox as LEtb

def match_zoom_wcs(axs, wcs_list, center_coord, RA_span, DEC_span):
    PA = Angle(np.arctan(RA_span.rad/DEC_span.rad),u.rad)
    Diag_span = Angle(np.hypot(RA_span.rad,DEC_span.rad),u.rad)
    corner1_world = center_coord.directional_offset_by(PA,Diag_span/2)
    corner2_world = center_coord.directional_offset_by((PA+Angle(180,u.deg)).wrap_at('360d'),Diag_span/2)
    for i in np.arange(len(axs)):
        # fig = figs[i]
        w = wcs_list[i]
        
        ax = axs[i]#fig.get_axes()[0]
        corner1_pix = corner1_world.to_pixel(w)
        corner2_pix = corner2_world.to_pixel(w)
        
        x_bnds = (corner1_pix[0], corner2_pix[0])
        y_bnds = (corner1_pix[1], corner2_pix[1])
        ax.set_xlim(np.min(x_bnds), np.max(x_bnds))
        ax.set_ylim(np.min(y_bnds), np.max(y_bnds))
    return

def remove_most_frequent(mat):
    mat = mat*1.0
    a,b=np.unique(mat.flatten(),return_counts=True)
    
    ind=np.argmax(b)
    value = a[ind]
    
    mat[mat==value] = np.nan
    return mat

def imshows(fitsDF, prof_sampDF_lst=None, plot_Fprofile=False, profDF=None, prof_crop = None, popts=None, FullScreen=False, fluxSpace='LIN',g_cl_arg=None, REF_image=None, med_filt_size=None, figs=None, peaks_locs=None, crest_lines=None):
    Zscale = ZScaleInterval()

    ax_img = []
    ax_img2 = []
    for ind in tqdm(np.arange(len(fitsDF))):
        HDU = fitsDF.iloc[ind]['Diff_HDU']
        w = fitsDF.iloc[ind]['WCS_w'].deepcopy()
        
        if figs is None:
            fig = plt.figure()
        else:
            fig = figs[ind]
        
        if prof_sampDF_lst is None:
            ax_img.append(fig.add_subplot(111,projection=w))
        else:
            ax_img.append(plt.subplot2grid((1,3),(0,0),rowspan=1,projection=w,fig=fig))
        
        mat = HDU.data*1.0
        clim = Zscale.get_limits(remove_most_frequent(mat))
        plt.imshow(mat,  vmin=clim[0], vmax=clim[1], origin='lower', cmap='gray', interpolation='none')

        if prof_sampDF_lst is not None:
            for i in np.arange(len(prof_sampDF_lst)):
                plt.sca(ax_img[ind])
                xy=prof_sampDF_lst[i].iloc[ind]['WrldCorners']
                if plot_Fprofile:
                    plt.plot(xy[:,0], xy[:,1], transform = ax_img[ind].get_transform('world'), linewidth=1)
                
                x=prof_sampDF_lst[i].iloc[ind]['ProjAng']
                y=prof_sampDF_lst[i].iloc[ind]['FluxProfile_ADU']
                ax_img2.append(plt.subplot2grid((1,3),(0,1),rowspan=1,colspan=2,fig=fig))
                plt.sca(ax_img2[ind])
                plt.scatter(x.arcsec,y,s=0.5, cmap='jet', label='flux samples')
                plt.xlabel('[arcsec]')
                plt.ylabel('flux [ADU]')
                plt.gca().legend()
    return ax_img