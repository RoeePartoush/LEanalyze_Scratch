#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:21:47 2020

@author: roeepartoush
"""

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


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def press(event):
    import warnings
    warnings.simplefilter('ignore')
    sys.stdout.flush()
    figures=[manager.canvas.figure for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
    
    fig_inds = []
    for fig in figures:
        fig_inds.append(fig.number)
    pchr = event.key
    if ((pchr=='z') | (pchr=='x')):
#        f_ind = int(pchr)-1
        if pchr=='z':
            inc = -1
        elif pchr=='x':
            inc = 1
        f_ind = np.mod(fig_inds.index(event.canvas.figure.number) +inc,len(figures))
        figures[f_ind].canvas.manager.window.activateWindow()
        figures[f_ind].canvas.manager.window.raise_()
#        figures[f_ind].canvas.draw()
        plt.figure(figures[f_ind].number)
    elif ((pchr=='a') | (pchr=='d')):
        for fig in figures:
            for ax in fig.get_axes():
                for im in ax.get_images():
                    cvmin, cvmax = im.get_clim()
                    inc = float(10)
                    if pchr=='a':
                        inc = -inc
                    elif pchr=='d':
                        inc = inc
                    im.set_clim(cvmin-inc, cvmax+inc)
            fig.canvas.draw()
    return

def onclick(event):
    global first_click
    global first_ax
    global g_DIFF_df
    global g_cl
    ix, iy = event.xdata, event.ydata
    if (ix is None) or (iy is None):
        return
    # w_ind = 0#event.inaxes.figure.number -1#plt.gcf().number -1
    # w=wcs_list[w_ind]
    indDF=np.argwhere((g_DIFF_df['filename']==event.inaxes.title.get_text()).to_numpy())[0,0]
    w = g_DIFF_df.iloc[indDF]['WCS_w']
    coord_tmp = Angle(w.wcs_pix2world(np.array([ix,iy],ndmin=2),0),u.deg)[0]
    #    print('FC: '+str(first_click))
    if first_click:
        first_ax = event.inaxes
        print('\n\n'+event.inaxes.title.get_text())
        print(indDF)
        g_DIFF_df.iloc[indDF]['coords_list'].append(np.array([coord_tmp[0].deg,coord_tmp[1].deg],ndmin=2))
        print('==================================\nPixel_1: x = %d, y = %d'%(ix, iy))
        
        print('World_1: RA = %.8f [deg], DEC = %.8f [deg]'%(coord_tmp[0].deg,coord_tmp[1].deg))
        global coord1
#        coord1 = coord_tmp
        coord1 = SkyCoord(coord_tmp[0],coord_tmp[1],frame='fk5')
        g_cl.append(np.array([coord_tmp[0].deg,coord_tmp[1].deg],ndmin=2))
#         first_click = False
#     else:#if event.inaxes==first_ax:
#         indDF1=np.argwhere((g_DIFF_df['filename']==first_ax.title.get_text()).to_numpy())[0,0]
#         indDF2=np.argwhere((g_DIFF_df['filename']==event.inaxes.title.get_text()).to_numpy())[0,0]
#         diff_mjd = g_DIFF_df.iloc[indDF2]['Idate'] - g_DIFF_df.iloc[indDF1]['Idate']
#         print('Pixel_2: x = %d, y = %d'%(ix, iy))
#         print('World_2: RA = %.8f [deg], DEC = %.8f [deg]'%(coord_tmp[0].deg,coord_tmp[1].deg))
#         first_click = True
# #        coord2 = coord_tmp
# #        [arcltPA, angSeper, mid_sph] = LeT.plot_arclt(coord2,coord1,Angle('0d1m0s'),event.inaxes,w)
#         coord2 = SkyCoord(coord_tmp[0],coord_tmp[1],frame='fk5')
#         arcltPA = (coord2.position_angle(coord1) + Angle(180,u.deg)).wrap_at(360 * u.deg)
#         angSeper = coord2.separation(coord1)
#         points=SkyCoord((coord1,coord2))
#         mid_sph=SkyCoord(points.data.mean(), representation_type='unitspherical', frame=points.frame)
#         plt.ioff()
#         for ax in [event.inaxes,first_ax]:
#             plt.sca(ax)
# #            print('kaka!')
#             plt.plot(np.array([coord1.ra.deg, coord2.ra.deg]), np.array([coord1.dec.deg, coord2.dec.deg]), color='c', transform = ax.get_transform('world'))
#             plt.scatter(coord2.ra.deg, coord2.dec.deg, s=20,color='m', transform = ax.get_transform('world'))
#             plt.scatter(coord1.ra.deg, coord1.dec.deg, s=20,color='g', transform = ax.get_transform('world'))
#             plt.scatter(mid_sph.ra.deg, mid_sph.dec.deg, s=20,color='y', transform = ax.get_transform('world'))
#         plt.ion()
#         print('World_mid: RA = %.8f [deg], DEC = %.8f [deg]'%(mid_sph.ra.deg,mid_sph.dec.deg))
#         print('PA: %.0f [deg]'%(arcltPA.deg))
#         print('Ang. sep.: '+str(angSeper.arcsec)+' [arcsec]')
#         print('Time sep.: '+str(diff_mjd)+' [days]')
#         if diff_mjd!=0:
#             print('Apparent motion: '+str(angSeper.arcsec/diff_mjd)+' [arcsec/day]')
#         print('==================================\n\n')
    return
def onpress(event):
    global press_bool
    press_bool=True
    return
def onmove(event):
    global press_bool, move
    if press_bool:
        move=True
    return
def onrelease(event):
    global press_bool, move
    if press_bool and not move:
        onclick(event)
    press_bool=False; move=False
    return

def im_norm_factor(HDU):
    try:
        factor = float(HDU.header['KSUM00'])
    except Exception as e:
        factor = 1.0
    return factor

def remove_most_frequent(mat):
    mat = mat*1.0
    a,b=np.unique(mat.flatten(),return_counts=True)
    
    ind=np.argmax(b)
    value = a[ind]
    #print(value)
    
    mat[mat==value] = np.nan
    return mat

def imshows(fitsDF, prof_sampDF_lst=None, plot_Fprofile=False, profDF=None, prof_crop = None, popts=None, FullScreen=False, fluxSpace='LIN',g_cl_arg=None, REF_image=None, med_filt_size=None, figs=None, peaks_locs=None, crest_lines=None):
    global press_bool, move, first_click
    press_bool, move, first_click = False, False, True
    global g_cl
    g_cl = g_cl_arg
    Zscale = ZScaleInterval()
    ax_img=[]
    ax2_img=[]
    ax3_img=[]
    global g_DIFF_df
    g_DIFF_df = fitsDF
    global wcs_list
    wcs_list = list()
    ylim_tmp = [np.inf, -np.inf]
    maxFWHM = Angle(list(fitsDF['FWHM_ang'])).max()
#    maxFWHM = maxFWHM+Angle(0.2,u.arcsec)
    last_mat = None
    hdr_R = fitsDF.iloc[0]['Diff_HDU'].header
    w_R = wcs.WCS(hdr_R)
    for ind in tqdm(np.arange(len(fitsDF))):
        if ind==len(fitsDF):
            last_mat = 1#REF_image
            print(ind)
            ind=ind-1
        HDU = fitsDF.iloc[ind]['Diff_HDU']
        # MaskMAT = fitsDF.iloc[ind]['Mask_mat']
        # NoiseMAT = fitsDF.iloc[ind]['Noise_mat']*1.0/float(HDU.header['KSUM00'])
        try:
            MaskMAT = fitsDF.iloc[ind]['Mask_HDU'].data
            NoiseMAT = fitsDF.iloc[ind]['Noise_HDU'].data
        except Exception as e:
            #print('\n***ERROR:*** '+str(e))
            MaskMAT = np.zeros(HDU.data.shape)
            NoiseMAT = np.zeros(HDU.data.shape)
        NoiseMAT = NoiseMAT*1.0/im_norm_factor(HDU)#float(HDU.header['KSUM00'])
        NoiseMAT[~LEtb.mask2bool(MaskMAT,mode='suspicious')] = np.nan
#        mat = NoiseMAT
        avg_noise = np.nanmean(NoiseMAT)
        # w = w_R
        w = fitsDF.iloc[ind]['WCS_w'].deepcopy()
        # w.wcs.crpix = w_R.wcs_world2pix(w.wcs_pix2world(w.wcs.crpix.reshape((1,2)),1).reshape((1,2)),1).reshape((2,))
#        w = wcs.WCS(HDU.header)
        
        if figs is None:
            fig = plt.figure()
        else:
            fig = figs[ind]
            #fig.clear()
        fig.canvas.mpl_connect('key_press_event', press)
#        global first_click, press_bool, move
#        first_click = True
#        press_bool=False
#        move = False
        fig.canvas.mpl_connect('button_press_event', onpress)
        fig.canvas.mpl_connect('button_release_event', onrelease)
        fig.canvas.mpl_connect('motion_notify_event', onmove)
        if FullScreen:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
        
        if prof_sampDF_lst is None:
            if ind==0:
                ax_img.append(fig.add_subplot(111,projection=w))
#                ax_img.append(fig.add_subplot(211,projection=w))
#                ax3_img.append(fig.add_subplot(212))
            else:
                # ax_img.append(fig.add_subplot(111,projection=w))
                ax_img.append(fig.add_subplot(111,projection=w))#,sharex=ax_img[0],sharey=ax_img[0]))
#                ax_img.append(fig.add_subplot(211,sharex=ax_img[0],sharey=ax_img[0],projection=w))
#                ax3_img.append(fig.add_subplot(212,sharex=ax3_img[0],sharey=ax3_img[0]))
        else:
            if ind==0:
                ax_img.append(plt.subplot2grid((1,3),(0,0),rowspan=1,projection=w,fig=fig))
#                ax2_img.append(plt.subplot2grid((3,1),(1,0),rowspan=2,projection='3d',fig=fig))#,facecolor='black'))
                # ax2_img.append(plt.subplot2grid((2,2),(0,1),rowspan=1,fig=fig))
                if fluxSpace=='MAG': 
                    ax2_img[ind].invert_yaxis()
            else:
                ax_img.append(plt.subplot2grid((1,3),(0,0),rowspan=1,projection=w,fig=fig))#,sharex=ax_img[0],sharey=ax_img[0]))
#                ax2_img.append(plt.subplot2grid((3,1),(1,0),rowspan=2,projection='3d',sharex=ax2_img[0],sharey=ax2_img[0],sharez=ax2_img[0]))#,facecolor='black',fig=fig))
                # ax2_img.append(plt.subplot2grid((2,2),(0,1),rowspan=1,sharex=ax2_img[0],fig=fig))
        
        # zptmag = 1#fitsDF.iloc[ind]['ZPTMAG']
        # zpt_lin = np.power(10,-zptmag/2.5)
        # sig_adu_cal = HDU.header['SKYSIG']*zpt_lin
        if REF_image is not None:
            mat = (LEtb.matchFWHM(fitsDF,ind,maxFWHM)*1.0/im_norm_factor(HDU))# - REF_image
            # mat = (LEtb.matchFWHM(fitsDF,ind,maxFWHM)*1.0/float(HDU.header['KSUM00'])) - REF_image
#            mat = (HDU.data*1.0/float(HDU.header['KSUM00'])) - REF_image
#            mat = HDU.data*1.0*zpt_lin - REF_image
        else:
            mat = HDU.data*1.0/im_norm_factor(HDU)
            # mat = HDU.data*1.0/float(HDU.header['KSUM00'])
        if med_filt_size is not None:
            mat = median_filter(mat,size=med_filt_size)
#            mat = mat[::med_filt_size,::med_filt_size]
#            MaskMAT = MaskMAT[::med_filt_size,::med_filt_size]
#            NoiseMAT = NoiseMAT[::med_filt_size,::med_filt_size]
#            w.wcs.crpix = w.wcs.crpix/med_filt_size
#            w.wcs.cd = w.wcs.cd*med_filt_size
#            w.array_shape = mat.shape
#        wcs_list.append(w)
##            mat = HDU.data*1.0*zpt_lin
##        mat = NoiseMAT*1.0/float(HDU.header['KSUM00'])#*zpt_lin
##        mat[mat<=-2*sig_adu_cal] = 0
#                
#         Zx = ndimage.sobel(mat,axis=1)/8
#         Zy = ndimage.sobel(mat,axis=0)/8
#         # grad = np.sqrt(Zx**2+Zy**2)
# #        grad_thres=np.percentile(grad,99.5)
# #        mat[grad>grad_thres] = np.nan
#         theta = np.arctan2(Zy,Zx)*180/np.pi
# #        mat[np.logical_or(np.abs(np.abs(theta)-90)<5,np.abs(np.abs(theta)-0)<5)] = np.nan
# #        mat = ndimage.filters.laplace(mat)
#         mat = theta
        mat[~LEtb.mask2bool(MaskMAT)] = np.nan
#        mat[~detect_peaks(NoiseMAT,MaskMAT)] = np.nan
        
#        if prof_sampDF_lst is None:
#            plt.sca(ax3_img[ind])
#            hist_vals, hist_cnts = np.unique(mat,return_counts=True)
#            hist_cnts_cum = np.cumsum(hist_cnts)
#            plt.plot(hist_vals,hist_cnts_cum)
##            plt.scatter(hist_vals[1:],np.divide(np.diff(hist_cnts_cum),np.diff(hist_vals)),s=1)
#    #        print('most common value: '+str(hist_vals[np.argmax(hist_cnts)]))
#    #        plt.scatter(np.argmin(np.abs(hist_cnts-0)))
#            plt.grid(color='black', ls='solid')
#            ax3_img[ind].axvline(x=avg_noise,color='m')
##            ax3_img[ind].axvline(x=np.nanmax(NoiseMAT),color='r')
##            ax3_img[ind].axvline(x=np.nanmin(NoiseMAT),color='b')
#            ax3_img[ind].axvline(x=-avg_noise,color='m')
        

#        mat[np.abs(mat)<30] = np.nan
#        mat[np.abs(mat)>300] = np.nan
#        mat[np.abs(mat)<(2*NoiseMAT/float(HDU.header['KSUM00']))] = np.nan
#        mat[mat<=0] = np.nan
#        print(np.sum(~np.isnan(mat)))
        clim = Zscale.get_limits(remove_most_frequent(mat))#[-5*sig_adu_cal,5*sig_adu_cal]#
        plt.sca(ax_img[ind])
        ax_img[ind].set_facecolor((0.5*135/255,0.5*206/255,0.5*235/255))
#        ax_img[ind].title.set_text(fitsDF.iloc[ind]['filename'])#+' / pipe_ver: '+str(fitsDF.iloc[ind]['pipe_ver']))
        # plt.gca().set_facecolor((0.5*135/255,0.5*206/255,0.5*235/255))
        plt.gca().title.set_text(fitsDF.iloc[ind]['filename'])#+' / pi
        if last_mat is not None:
            print('HEYYYY!')
            print(ind)
            print(mat.shape)
            print('max:'+str(np.nanmax(mat))+', min:'+str(np.nanmin(mat)))
            mat = REF_image
        # im = ax_img[ind].imshow(mat,  origin='lower', cmap='gray', interpolation='none')
        plt.imshow(mat,  vmin=clim[0], vmax=clim[1], origin='lower', cmap='gray', interpolation='none')#,extent=get_pix_extent(hdr_R,HDU.header))
        # plt.imshow(mat, vmin=clim[0], vmax=clim[1], origin='lower', cmap='gray', interpolation='none')        
        # a = np.eye(3)
        # a[0:2,0:2] = w.wcs.cd / np.sqrt(np.abs(np.linalg.det(w.wcs.cd)))
        # a_trans = mtransforms.Affine2D(a)
        # b = np.eye(3)
        # b[0:2,0:2] = w_R.wcs.cd / np.sqrt(np.abs(np.linalg.det(w_R.wcs.cd)))
        # b_trans = mtransforms.Affine2D(b)
        # im_trans = a_trans.inverted() + b_trans
        # trans = im_trans + ax_img[ind].transData#(a_trans-mtransforms.Affine2D(b))
        # # ex_trans = ax_img[ind].transData.inverted() + im_trans
        # # trans = ax_img[ind].transData + (mtransforms.Affine2D(w.wcs.cd)-mtransforms.Affine2D(w_R.wcs.cd))
        # print(ax_img[ind].transData.get_matrix())
        # print(trans.get_matrix())
        # extent = get_pix_extent(hdr_R,HDU.header)
        # print(extent)
        # corner1 = im_trans.inverted().transform(np.array([extent[0],extent[2]]))
        # corner2 = im_trans.inverted().transform(np.array([extent[1],extent[3]]))
        # print((corner1[0],corner2[0],corner1[1],corner2[1]))
        # im.set_extent((corner1[0],corner2[0],corner1[1],corner2[1]))
        # im.set_transform(trans)
        plt.grid(color='white', ls='solid')
        
#        imshow_Xcorr(fitsDF, 0, ind)
#        _=plt.hist(mat.flatten(),1000,log=True)
#        plt.imshow(np.fft.fftshift(np.log10(np.abs(np.fft.fft2(HDU.data*1.0)))),cmap='jet')
#        plt.imshow(np.fft.fftshift(np.angle(np.fft.fft2(HDU.data*1.0))),cmap='hsv')
        
        if prof_sampDF_lst is not None:
#            ylim_tmp = [np.inf, -np.inf]
            tmp_sct=[]
            for i in np.arange(len(prof_sampDF_lst)):
                ref_FP = 0#*np.nanmedian(np.stack(prof_sampDF_lst[i]['FP_Ac_bin_y'].to_numpy()),axis=0)
#                ref_FP[np.isnan(ref_FP)]=0
                plt.sca(ax_img[ind])
                xy=prof_sampDF_lst[i].iloc[ind]['WrldCorners']
                if plot_Fprofile:
                    plt.plot(xy[:,0], xy[:,1], transform = ax_img[ind].get_transform('world'), linewidth=1)
                
                prof_orig = profDF['Orig'][0]
                prof_PA_x = profDF['PA'][0]
                prof_PA_y = profDF['PA'][0] + Angle(90,u.deg)
                
                if prof_crop:
                    x_lims = Angle(prof_crop[0][ind],u.arcsec)
                    y_lims = Angle(prof_crop[1][ind],u.arcsec)
                    corner1 = prof_orig.directional_offset_by(prof_PA_x,x_lims[0]).directional_offset_by(prof_PA_y,y_lims[0])
                    corner2 = prof_orig.directional_offset_by(prof_PA_x,x_lims[1]).directional_offset_by(prof_PA_y,y_lims[0])
                    corner3 = prof_orig.directional_offset_by(prof_PA_x,x_lims[1]).directional_offset_by(prof_PA_y,y_lims[1])
                    corner4 = prof_orig.directional_offset_by(prof_PA_x,x_lims[0]).directional_offset_by(prof_PA_y,y_lims[1])
                    xy = np.array([[corner1.ra.degree, corner1.dec.degree], [corner2.ra.degree, corner2.dec.degree], 
                                   [corner3.ra.degree, corner3.dec.degree], [corner4.ra.degree, corner4.dec.degree], [corner1.ra.degree, corner1.dec.degree]])
                    plt.plot(xy[:,0], xy[:,1], transform = ax_img[ind].get_transform('world'), linewidth=1, color='b')
                
                if popts:
                    prof_width = profDF['WIDTH'][0]
                    peak_arcsec_shift = Angle(popts[ind][2],u.arcsec)
                    corner1 = prof_orig.directional_offset_by(prof_PA_x,peak_arcsec_shift).directional_offset_by(prof_PA_y,-prof_width/2)
                    corner2 = prof_orig.directional_offset_by(prof_PA_x,peak_arcsec_shift).directional_offset_by(prof_PA_y, prof_width/2)
                    xy = np.array([[corner1.ra.degree, corner1.dec.degree], [corner2.ra.degree, corner2.dec.degree]])
                    plt.plot(xy[:,0], xy[:,1], transform = ax_img[ind].get_transform('world'), linewidth=0.5, color='r')
                
                x=prof_sampDF_lst[i].iloc[ind]['ProjAng']
                
#                if REF_prof_sampDF_lst is not None
                xx,yy,yyerr, yBref, stdBref, binCnt = prof_sampDF_lst[i].iloc[ind][['FP_Ac_bin_x','FP_Ac_bin_y','FP_Ac_bin_yerr','FP_Ac_bin_yBref','FP_Ac_bin_ystdBref','FP_Ac_bin_Cnt']].to_list()
                # yy=None
                if yy is not None:
#                    xy=prof_sampDF_lst[i].iloc[ind]['WrldVec']
#                    plt.sca(ax_img[ind])
#                    plt.scatter(xy[:,0], xy[:,1], s=1,color='r', transform = ax_img[ind].get_transform('world'))
                    plt.sca(ax2_img[ind])
#                    yy[np.isnan(yy)]=0
#                    plt.plot(xx,ndimage.gaussian_filter(yy,sigma=10/2*np.sqrt(2*np.log(2))))
                    plt.errorbar(xx,yy-ref_FP,yerr=yyerr,label='binned flux profile')#,fmt='.b')
#                    ax2_img[ind].axhline(y=avg_noise/np.mean(np.sqrt(binCnt)))
                    if fluxSpace=='LIN':
                        y=prof_sampDF_lst[i].iloc[ind]['FluxProfile_ADUcal']#['NoiseProfile_ADU']/float(HDU.header['KSUM00'])#
                    elif fluxSpace=='MAG': 
                        y=prof_sampDF_lst[i].iloc[ind]['FluxProfile_MAG']
                    elif fluxSpace=='LIN-ADU': 
                        y=prof_sampDF_lst[i].iloc[ind]['FluxProfile_ADU']
                    
                    meansig = np.nanmean(np.abs(yyerr))
#                    tmp_sct.append(plt.scatter(xx,np.ones(xx.shape)*i*2,s=1,c=yy,cmap='jet'))
#                    plt.scatter(prof_sampDF_lst[i].iloc[ind]['Peak_ProjAng_arcsec'],i*2,s=2,c='m',cmap='jet')
                    # ylim_tmp = [min(ylim_tmp[0],np.nanmin(yy)-3*meansig), max(ylim_tmp[1],np.nanmax(yy)+3*meansig)]
                    # ylim_tmp = [np.nanpercentile(yy, 0)-3*meansig, np.nanpercentile(yy, 90)+3*meansig]
                    # plt.plot(x.arcsec[x.arcsec.argsort()],y[x.arcsec.argsort()],linewidth=0.2,label='flux samples')
                    plt.scatter(x.arcsec,y,s=0.5, cmap='jet', label='flux samples')#,c=np.abs((y-yBref)/stdBref))
#                    yyerr_skysig = np.zeros(yyerr.shape)
#                    yyerr_skysig = np.divide(sig_adu_cal,np.sqrt(binCnt))
#                    plt.plot(xx,np.divide(yyerr,yyerr_skysig))
#    #                print(ylim_tmp)
#                    ax2_img[ind].scatter(xx,np.ones(xx.shape)*i,yy,s=1)
#                    ax2_img[ind].scatter(x.arcsec,np.ones(x.shape)*i,y,s=0.1,c=np.abs((y-yBref)/stdBref),vmax=2,vmin=0, cmap='jet')
#                    plt.xlabel('x')
#                    plt.ylabel('y')
                    # ax2_img[ind].set_ylim(ylim_tmp[0],ylim_tmp[1])
                    plt.xlabel('[arcsec]')
                    plt.ylabel('flux [ADU]')
                    plt.gca().legend()
                    if peaks_locs is not None:
                        peak_offset = peaks_locs[ind,1]
                        peak_err = peaks_locs[ind,2]
                        # ax2_img[ind].axvline(x=peak_offset,color='r')
                        # ax2_img[ind].axvline(x=peak_offset+peak_err,color='orange')
                        # ax2_img[ind].axvline(x=peak_offset-peak_err,color='orange')
#                    plt.ylim(np.percentile(y,1),np.percentile(y,99.999))
#            plt.gca().set_aspect('equal', adjustable='box')
#            for i in np.arange(len(prof_sampDF_lst)):
#                tmp_sct[i].set_clim(ylim_tmp[0]*0,ylim_tmp[1])
#                tmp_sct[i].set_facecolor((0,0,0))
        if profDF is not None: 
            plt.sca(ax_img[ind])
            for s_coord in profDF['Orig']:
                plt.scatter(s_coord.ra.degree, s_coord.dec.degree, s=20,color='b', transform = ax_img[ind].get_transform('world'))
                slit_PA = profDF.iloc[0]['PA']
                slit_PA_perp = Angle(profDF.iloc[0]['PA'] + Angle(90,u.deg)).wrap_at('360d')
                if s_coord is profDF.iloc[0]['Orig']:
                    if peaks_locs is not None:
                        peak_offset = peaks_locs[ind,1]
                        peak_err = peaks_locs[ind,2]
                        
                        peak_coord = s_coord.directional_offset_by(slit_PA,Angle(peak_offset,u.arcsec))
                        plt.scatter(peak_coord.ra.degree, peak_coord.dec.degree, s=20,color='r', transform = ax_img[ind].get_transform('world'))
                        
                        peak_err1 = peak_coord.directional_offset_by(slit_PA,Angle(peak_err,u.arcsec))
                        peak_err2 = peak_coord.directional_offset_by((slit_PA+Angle(180,'deg')).wrap_at('360d'),Angle(peak_err,u.arcsec))
                        peak_err = SkyCoord((peak_err1,peak_err2))
                        plt.plot(peak_err.ra.degree, peak_err.dec.degree,color='r', transform = ax_img[ind].get_transform('world'))
                    if crest_lines is not None:
                        x1, y1 = crest_lines[ind,0,0:2]
                        x2, y2 = crest_lines[ind,1,0:2]
                        crest_line1 = s_coord.directional_offset_by(slit_PA,Angle(x1,u.arcsec)).directional_offset_by(slit_PA_perp,Angle(y1,u.arcsec))
                        crest_line2 = s_coord.directional_offset_by(slit_PA,Angle(x2,u.arcsec)).directional_offset_by(slit_PA_perp,Angle(y2,u.arcsec))
                        crest_line = SkyCoord((crest_line1,crest_line2))
                        plt.plot(crest_line.ra.degree, crest_line.dec.degree,color='r', transform = ax_img[ind].get_transform('world'))
                    
                    # x1, y1 = -9,0
                    # x2, y2 = 9,0
                    # crest_line1 = s_coord.directional_offset_by(slit_PA,Angle(x1,u.arcsec)).directional_offset_by(slit_PA_perp,Angle(y1,u.arcsec))
                    # crest_line2 = s_coord.directional_offset_by(slit_PA,Angle(x2,u.arcsec)).directional_offset_by(slit_PA_perp,Angle(y2,u.arcsec))
                    # crest_line = SkyCoord((crest_line1,crest_line2))
                    # plt.plot(crest_line.ra.degree, crest_line.dec.degree,color='b', transform = ax_img[ind].get_transform('world'))
    return ax_img

def imshow_Xcorr(fitsDF, ind1, ind2):
    Zscale = ZScaleInterval()
    fig=plt.figure()
    fig.canvas.mpl_connect('key_press_event', press)
    mat1_F = np.fft.fft2(get_mat(fitsDF, ind1, conf='Xcorr'))
    mat2_F = np.fft.fft2(get_mat(fitsDF, ind2, conf='Xcorr'))
    
    xcor = np.fft.fftshift(np.fft.ifft2(np.multiply(np.conj(mat1_F),mat2_F)).real)
    clim = Zscale.get_limits(xcor)
    plt.imshow(xcor, origin='lower', cmap='gray', vmin=0*clim[0], vmax=clim[1], interpolation='none')
    
    return

def light_curve(fitsDF, s_coords):
    x = np.zeros((len(fitsDF),))
    y = np.zeros((len(fitsDF),len(s_coords)))
    plt.figure()
    for i in np.arange(len(fitsDF)):
        x[i] = fitsDF.iloc[i]['Idate']
        HDU = fitsDF.iloc[i]['Diff_HDU']
        w = wcs.WCS(HDU.header)
        zpt_lin = np.power(10,-fitsDF.iloc[i]['ZPTMAG']/2.5)
        sc_ind = 0
        for sc in s_coords:
            sc_pix = w.wcs_world2pix(np.array([[sc.ra.deg, sc.dec.deg]]), 0)
            y[i,sc_ind] = HDU.data[sc_pix[0,1].round().astype(int),sc_pix[0,0].round().astype(int)]*zpt_lin
            sc_ind = sc_ind+1
    
    for i in np.arange(len(s_coords)):
        plt.scatter(x,y[:,i],s=1)
    
    return
def get_mat(fitsDF, ind, conf='Xcorr'):
    
    mat = fitsDF.iloc[ind]['Diff_HDU'].data*1.0
    MaskMAT = fitsDF.iloc[ind]['Mask_mat']
    
    if conf=='Xcorr':
        fill_val = np.abs(mat).min()
        mat[~LEtb.mask2bool(MaskMAT)] = fill_val
#        mat[mat<=0] = fill_val
        
    return mat


def detect_peaks(image,mask,data_mat=None,Nbins=1000):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    if data_mat is None:
        data_mat = np.ones(image.shape)
    data_mat = data_mat.copy()
    mask_safe = LEtb.mask2bool(mask,'safe')
    peak_x,FWHM = estdist(image[mask_safe],Nbins)
    mask_sus_o = LEtb.mask2bool(mask,'suspicious-only')
    peak_x_sus_o,FWHM_sus_o = estdist(image[mask_sus_o],Nbins)
    mask_sus = LEtb.mask2bool(mask,'suspicious')

    
    image_c = image.copy()
    image_c[~mask_sus] = 0
    image_c[np.logical_and(image<(peak_x+5*FWHM),mask_safe)] = 0#np.nan
#    image_c[np.logical_and(image<(peak_x+5*FWHM),mask_sus_o)] = 0
    image_c[np.logical_and(image<(peak_x_sus_o+5*FWHM_sus_o),mask_sus_o)] = 0
#    ax1=plt.figure().gca()
#    plt.imshow(image,cmap='gray',vmin=0,vmax=15)
#    plt.figure().add_subplot(111,sharex=ax1,sharey=ax1)
    star_mask = ~np.logical_or(image_c!=0,~mask_sus)
#    plt.imshow(star_mask*1.0,cmap='gray')#,vmin=0,vmax=15)
    
    Zscale = ZScaleInterval()
    clim = Zscale.get_limits(data_mat)
#    plt.figure().add_subplot(111,sharex=ax1,sharey=ax1)
    data_mat[~star_mask] = 0
#    plt.imshow(data_mat,cmap='gray',vmin=clim[0],vmax=clim[1])
    
#    # define an 8-connected neighborhood
#    neighborhood = generate_binary_structure(2,2)
#    #apply the local maximum filter; all pixel of maximal value 
#    #in their neighborhood are set to 1
#    local_max = maximum_filter(image, size=10)==image
##    local_max = maximum_filter(image, footprint=neighborhood)==image
#    #local_max is a mask that contains the peaks we are 
#    #looking for, but also the background.
#    #In order to isolate the peaks we must remove the background from the mask.
#
#    #we create the mask of the background
#    background = (image==0)
#
#    #a little technicality: we must erode the background in order to 
#    #successfully subtract it form local_max, otherwise a line will 
#    #appear along the background border (artifact of the local maximum filter)
#    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
#
#    #we obtain the final mask, containing only peaks, 
#    #by removing the background from the local_max mask (xor operation)
#    detected_peaks = local_max ^ eroded_background

    return star_mask#detected_peaks


def estdist(image,Nbins):
    hist_vals, hist_cnts = np.unique(image,return_counts=True)
    hist_cnts_cum = np.cumsum(hist_cnts)
    rng_min = hist_vals[np.nanargmin(np.abs(hist_cnts_cum-(hist_cnts_cum[-1]*0.01)))]
#    print(rng_min)
    rng_max = hist_vals[np.nanargmin(np.abs(hist_cnts_cum-(hist_cnts_cum[-1]*0.99)))]
#    print(rng_max)
    statistic, bin_edges, _ = stats.binned_statistic(hist_vals,hist_cnts_cum/hist_cnts_cum[-1],statistic='mean',bins=Nbins,range=(rng_min,rng_max))
    bin_cntrs = (bin_edges[:-1]+bin_edges[1:])/2
    statistic = np.interp(bin_cntrs,bin_cntrs[~np.isnan(statistic)],statistic[~np.isnan(statistic)])
    pdf_y = np.diff(statistic)/np.diff(bin_cntrs)
    pdf_x = bin_edges[1:-1]
#    plt.figure()
#    plt.plot(hist_vals,hist_cnts_cum)
#    plt.scatter(pdf_x,pdf_y,s=1,c='r')
    pdf_y = ndimage.gaussian_filter(pdf_y,sigma=5/2*np.sqrt(2*np.log(2)))
    peak_x = pdf_x[np.nanargmax(pdf_y)]
    ind_FWHM = np.abs(pdf_y-0.5*np.nanmax(pdf_y)).argmin()
    FWHM = np.abs(peak_x-pdf_x[ind_FWHM])
#    plt.scatter(pdf_x,pdf_y,s=1)
#    plt.grid(color='black', ls='solid')
#    plt.gca().axvline(x=peak_x,color='m')
#    plt.gca().axvline(x=peak_x+FWHM,color='g')
#    plt.gca().axvline(x=peak_x-FWHM,color='g')
#    plt.gca().axvline(x=peak_x+5*FWHM,color='r')
#    plt.gca().axvline(x=peak_x-5*FWHM,color='r')
#    plt.gca().axhline(y=pdf_y[ind_FWHM],color='g')
    return peak_x, FWHM

def imshows_shifted(fitsDF,PA,app_mot,ref_ind=None,med_filt_size=None,share_ax=None,plot_bl=True,downsamp=False):
    if ref_ind is None:
        ref_ind = int(len(fitsDF)/2)
    Zscale = ZScaleInterval()
    ax_img=[]
    mat_lst=[]
    w_lst=[]
    # SN_sc = SkyCoord('23h23m24s','+58°48.9′',frame='fk5') # Cas A
    SN_sc = SkyCoord('0h25.3m','+64° 09′',frame='fk5') # Tycho
    for ind in tqdm(np.arange(len(fitsDF))):
        HDU = fitsDF.iloc[ind]['Diff_HDU']
#        MaskMAT = fitsDF.iloc[ind]['Mask_mat']
        mat = HDU.data*1.0/im_norm_factor(HDU)
        if med_filt_size is not None:
            mat = median_filter(mat,size=med_filt_size)
        w = wcs.WCS(HDU.header)
        diff_mjd = (fitsDF.iloc[ind]['Idate'] - fitsDF.iloc[ref_ind]['Idate'])*u.day
        FPcntr_world = w.wcs_pix2world(np.array([mat.shape[1],mat.shape[1]],ndmin=2)/2, 0)
        ra = Longitude(FPcntr_world[0,0],unit=u.deg)
        dec = Latitude(FPcntr_world[0,1],unit=u.deg)
        LE_sc = SkyCoord(ra, dec, frame='fk5')
        if PA is None:
            PA = LE_sc.position_angle(SN_sc)+Angle(180,'deg')
        _, FP_pa_pix_uv, FPlen_pix = LEtb.FPparams_world2pix(w, LE_sc, PA, Angle(-app_mot*diff_mjd))
        shift = np.squeeze(FPlen_pix*FP_pa_pix_uv)
#        print(np.round(shift))
        mat=ndimage.shift(mat,np.round(np.array([shift[1], shift[0]])),order=0)
        w = WCS_shift(w,shift)
        if downsamp and (med_filt_size is not None):
            w = WCS_downsample(w,med_filt_size,mat.shape)
            mat = mat[::med_filt_size,::med_filt_size]
        if plot_bl:
            fig = plt.figure()
            fig.canvas.mpl_connect('key_press_event', press)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            if ind==0:
                if share_ax is None:
                    ax_img.append(fig.add_subplot(111,projection=w))
                else:
                    ax_img.append(fig.add_subplot(111,projection=w,sharex=share_ax,sharey=share_ax))
            else:
                ax_img.append(fig.add_subplot(111,sharex=ax_img[0],sharey=ax_img[0],projection=w))
            clim = Zscale.get_limits(mat)
            plt.imshow(mat, vmin=clim[0], vmax=clim[1], origin='lower', cmap='gray', interpolation='none')
            plt.grid(color='white', ls='solid')
            plt.title('App. motion: '+str(app_mot))
        else:
            mat_lst.append(mat)
            w_lst.append(w)
    if not plot_bl:
        return mat_lst, w_lst#np.stack(mat_lst,axis=2)

def calc_shifts(fitsDF,PA,app_mot,ref_ind=None):
    if ref_ind is None:
        ref_ind = int(len(fitsDF)/2)
    shifts_lst = []
    # SN_sc = SkyCoord('23h23m24s','+58°48.9′',frame='fk5') # Cas A
    SN_sc = SkyCoord('0h25.3m','+64° 09′',frame='fk5') # Tycho
    for ind in tqdm(np.arange(len(fitsDF))):
        HDU = fitsDF.iloc[ind]['Diff_HDU']
        w = wcs.WCS(HDU.header)
        diff_mjd = (fitsDF.iloc[ind]['Idate'] - fitsDF.iloc[ref_ind]['Idate'])*u.day
        FPcntr_world = w.wcs_pix2world(np.array([w.array_shape[0],w.array_shape[1]],ndmin=2)/2, 0)
        ra = Longitude(FPcntr_world[0,0],unit=u.deg)
        dec = Latitude(FPcntr_world[0,1],unit=u.deg)
        LE_sc = SkyCoord(ra, dec, frame='fk5')
        if PA is None:
            PA = LE_sc.position_angle(SN_sc)+Angle(180,'deg')
        _, FP_pa_pix_uv, FPlen_pix = LEtb.FPparams_world2pix(w, LE_sc, PA, Angle(-app_mot*diff_mjd))
        shift = np.squeeze(FPlen_pix*FP_pa_pix_uv) # shift ~ ([x],[y])
        shifts_lst.append(shift)
    return shifts_lst


def WCS_downsample(w,N,shape,inplace=False):
    if inplace:
        w2 = w
    else:
        w2 = w.deepcopy()
    w2.wcs.crpix = w.wcs.crpix/N
    w2.wcs.cd = w.wcs.cd*N
    w2.array_shape = shape
    if not inplace:
        return w2

def WCS_shift(w,shift_pix,inplace=False):
    if inplace:
        w.wcs.crpix = w.wcs.crpix + shift_pix
    else:
        w2 = w.deepcopy()
        w2.wcs.crpix = w2.wcs.crpix + shift_pix
        return w2


class IndexTracker(object):
    def __init__(self, ax, X, titles, w_lst, shifts):
        self.shifts = shifts
        self.Zscale = ZScaleInterval()
        self.titles = titles
        self.w_lst = w_lst
        self.ind = 0
        self.w_0 = self.w_lst[0].deepcopy()
        self.curr_w = self.w_lst[self.ind].deepcopy()
        self.X = self.X2list(X)
        curr_mat = self.X[self.ind] # X[:,:,self.ind]
        self.clim = self.Zscale.get_limits(self.remove_most_frequent(curr_mat))
        if ax is not None:
            self.ax = ax
        else:
            self.ax = plt.subplot(111,projection=self.curr_w)
        self.ax.figure.canvas.mpl_connect('key_press_event', self.onpress)
        self.slices = len(self.X)
        self.im = self.ax.imshow(curr_mat,vmin=self.clim[0],vmax=self.clim[1],cmap='gray')
        self.im.set_extent(get_pix_extent(self.w_0,self.curr_w))
        plt.xlim(0,self.curr_w.array_shape[1])
        plt.ylim(0,self.curr_w.array_shape[0])
        self.ax.grid(color='white', ls='solid')
        self.PixelTrans = wcs.WCS()#transforms.Affine2D()
        self.overlay = self.ax.get_coords_overlay(self.PixelTrans)
        self.overlay.grid(color='black', linestyle='solid', alpha=0.5)
        self.update()
    
    def X2list(self,X):
        if isinstance(X,list):
            return X
        elif isinstance(X,np.ndarray):
            X_lst = []
            for i in np.arange(X.shape[2]):
                X_lst.append(X[:,:,i])
            return X_lst
        else:
            print('IndexTracker received neither a 3D np.ndarray nor a list of images, but a '+str(type(X)))
            return None
    
    def remove_most_frequent(self,mat):
        mat = mat*1.0
        a,b=np.unique(mat.flatten(),return_counts=True)
        
        ind=np.argmax(b)
        value = a[ind]
                
        mat[mat==value] = np.nan
        return mat
    
    def onscroll(self, event):
#        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()
    
    def onpress(self, event):
#        self.gl_ev.append(event)
#        f_ind = np.mod(fig_inds.index(plt.gcf().number) +inc,len(figures))
#        figures[f_ind].canvas.manager.window.activateWindow()
#        figures[f_ind].canvas.manager.window.raise_()
#        print("%s %s" % (event.button, event.step))
        if event.key == 'up':
            self.ind = (self.ind + 1) % self.slices
        elif event.key == 'down':
            self.ind = (self.ind - 1) % self.slices
        elif event.key=='z' or event.key=='x':
            managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
            fig_nums = [manager.canvas.figure.number for manager in managers]
            curr_fig_ind = fig_nums.index(event.canvas.figure.number)
            if event.key == 'z':
                next_fig_ind = (curr_fig_ind - 1) % len(managers)
#                next_fig_ind = (event.canvas.figure.number-1 - 1) % len(managers)
            elif event.key == 'x':
                next_fig_ind = (curr_fig_ind + 1) % len(managers)
#                next_fig_ind = (event.canvas.figure.number-1 + 1) % len(managers)
            managers[next_fig_ind].window.activateWindow()
            managers[next_fig_ind].window.raise_()
        self.update()

    def update(self):
        curr_mat = self.X[self.ind]
        self.clim = list(self.Zscale.get_limits(self.remove_most_frequent(curr_mat)))
        span=self.clim[1]-self.clim[0]
        self.clim[1] = self.clim[1]+1*span
#        self.ax = plt.subplot(111,projection=self.w_lst[self.ind])
#        self.im = ax.imshow(self.X[:, :, self.ind],vmin=self.clim[0],vmax=self.clim[1],cmap='gray')
#        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.set_data(curr_mat)
        self.im.set_clim(self.clim[0],self.clim[1])
        self.ax.set_title(self.titles[self.ind])
        w_n = self.w_lst[self.ind].deepcopy()
        new_extent = get_pix_extent_shifted(self.w_0, w_n, self.shifts[self.ind])
        self.im.set_extent(new_extent)
        # print(new_extent)
        self.curr_w.wcs.crpix = self.w_0.wcs.crpix + pix2pix_vec(self.w_0, w_n, self.shifts[self.ind]) - pix2pix_vec(self.w_0, w_n, np.array([0,0]))# = self.w_0.wcs_world2pix(w_n.wcs_pix2world(w_n.wcs.crpix.reshape((1,2)),0),0).flatten()
#        self.ax.set_transform(self.w_lst[self.ind])
#        self.ax.grid(color='white', ls='solid')
        # plt.xlim(0,self.curr_w.array_shape[1])
        # plt.ylim(0,self.curr_w.array_shape[0])
        self.im.axes.figure.canvas.draw()

def imshowMat(mat,w=None,shareaxFig=None):
    Zscale = ZScaleInterval()
    fig = plt.figure()
    if w is not None:
        if shareaxFig is not None:
            fig.add_subplot(111,projection=w,sharex=shareaxFig.get_axes()[0],sharey=shareaxFig.get_axes()[0])
        else:
            fig.add_subplot(111,projection=w)
        plt.grid(color='white', ls='solid')
    clim = Zscale.get_limits(remove_most_frequent(mat))
    plt.imshow(mat, vmin=clim[0], vmax=clim[1], origin='lower', cmap='gray', interpolation='none')
    return

def get_pix_extent(WoH_r,WoH_n): # WCS or HEADER of reference / new, respectively
    wcs_r = WoH2W(WoH_r)
    wcs_n = WoH2W(WoH_n)
    nrows_n = wcs_n.array_shape[0]#hdr_n['NAXIS2']
    ncols_n = wcs_n.array_shape[1]#hdr_n['NAXIS1']
    
    corner1 = wcs_r.wcs_world2pix(np.array(wcs_n.wcs_pix2world(0,0,0),ndmin=2),0)
    corner2 = wcs_r.wcs_world2pix(np.array(wcs_n.wcs_pix2world(ncols_n,nrows_n,0),ndmin=2),0)
    extent = (corner1[0,0], corner2[0,0], corner1[0,1], corner2[0,1])
    return extent


def get_pix_extent_shifted(WoH_r,WoH_n,shift): # WCS or HEADER of reference / new, respectively
    wcs_r = WoH2W(WoH_r)
    wcs_n = WoH2W(WoH_n)
    nrows_n = wcs_n.array_shape[0]#hdr_n['NAXIS2']
    ncols_n = wcs_n.array_shape[1]#hdr_n['NAXIS1']
    
    corner1 = pix2pix_vec(wcs_r, wcs_n, np.array([0,0])+shift)
    corner2 = pix2pix_vec(wcs_r, wcs_n, np.array([ncols_n,nrows_n])+shift)
    extent = (corner1[0], corner2[0], corner1[1], corner2[1])
    return extent

def pix2pix_vec(wcs_r, wcs_n, vec_n):
    vec_r = wcs_r.wcs_world2pix(np.array(wcs_n.wcs_pix2world(vec_n[0],vec_n[1],0),ndmin=2),0).reshape((2,))
    return vec_r

def WoH2W(WoH):
    if isinstance(WoH,wcs.WCS):
        return WoH
    elif isinstance(WoH,fits.header.Header):
        return wcs.WCS(WoH)
    else:
        print('WoH2W received non WCS-able object of type '+str(type(WoH)))
        return None

def imshow_copy_figure(source_fig, target_fig, source_im_ind=0, alpha=1, alpha_adj=False):
    source_im = source_fig.get_axes()[0].get_images()[source_im_ind]
    mat = source_im.get_array()
    extent = source_im.get_extent()
    clim = ZScaleInterval().get_limits(mat)
    target_ax = target_fig.get_axes()[0]
    # target_ax.get_images()[0].set_alpha(1-alpha)
    target_ax.imshow(mat,vmin=clim[0],vmax=clim[1],cmap='gray',extent=extent, alpha=alpha)
    return

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

