#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:42:18 2019

@author: roeeyairpartoush
"""
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.axes as pltax

import scipy.interpolate as spi
from scipy.ndimage import gaussian_filter

import pandas as pd

from astropy.io import fits
from astropy import wcs
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.visualization import ZScaleInterval
#from astropy.wcs.utils import wcs_to_celestial_frame
#from astropy.coordinates import SkyCoord

from PIL import Image

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
        f_ind = np.mod(fig_inds.index(plt.gcf().number) +inc,len(figures))
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

#def onclick(event):
##    global ix, iy
#    ix, iy = event.xdata, event.ydata
#    w_ind = plt.gcf().number -1
#    w=wcs_list[w_ind]
#    
#    coord_tmp = Angle(w.wcs_pix2world(np.array([ix,iy],ndmin=2),0),u.deg)[0]
#    
#    global first_click
#    print('FC: '+str(first_click))
#    if first_click:
#        print('\n\n==================================\nPixel_1: x = %d, y = %d'%(ix, iy))
#        print('World_1: RA = %.8f [deg], DEC = %.8f [deg]'%(coord_tmp[0].deg,coord_tmp[1].deg))
#        global coord1
#        coord1 = coord_tmp    
#        first_click = False
#    else:
#        print('Pixel_2: x = %d, y = %d'%(ix, iy))
#        print('World_2: RA = %.8f [deg], DEC = %.8f [deg]'%(coord_tmp[0].deg,coord_tmp[1].deg))
#        first_click = True
#        coord2 = coord_tmp
#        [arcltPA, angSeper] = plot_arclt(coord2,coord1,Angle('0d1m0s'),plt.gca(),w)
#        print('PA: '+str(arcltPA.deg)+' [deg]')
#        print('Ang. sep.: '+str(angSeper.arcmin)+' [arcmin]')
#        print('==================================\n\n')
    
    return
# =============== ang2pix_scale ================
def ang2pix_scale(ang_len, w):
    P2W = w.wcs_pix2world
    W2P = w.wcs_world2pix
    SQ = np.squeeze
    origpix = np.array([0,0],ndmin=2)
    SEwrld0 = P2W(origpix,0)
    SElnlenwrld = np.array([SEwrld0[0][0], ang_len.degree + SEwrld0[0][1]],ndmin=2)
    SElnlenpix = W2P(SElnlenwrld,0)
    pix_vec = SQ(SElnlenpix - origpix)
    pix_len = v_norma(pix_vec)
    return pix_len

# =============== ang2pix_scale ================
def pix2ang_scale(pix_len, w):
    P2W = w.wcs_pix2world
    W2P = w.wcs_world2pix
    SQ = np.squeeze
    origpix = np.array([0,0],ndmin=2)
    SEwrld0 = P2W(origpix,0)
    SElnlenpix = np.array([origpix[0][0], pix_len + origpix[0][1]],ndmin=2)
    SElnlenwrld = P2W(SElnlenpix,0)
    ang_vec = SQ(SElnlenwrld - SEwrld0)
    ang_len = Angle(v_norma(ang_vec),'degree')
    return ang_len


def vectanpth2arcsec(smp_vec,w,cntrPix,uvPix):
    SQ = np.squeeze
    axind = int(np.argwhere(np.logical_not(np.array(smp_vec.shape)==2)))
    N = smp_vec.shape[axind]
    arcsc_ln = np.zeros((N,))
    smp_vecRS = np.reshape(smp_vec,(N,2))
    cntrPixRS = SQ(cntrPix)#np.reshape(cntrPix,(1,2))
    uvPixRS = SQ(uvPix)#np.reshape(uvPix,(1,2))
#    print(smp_vecRS.shape)
#    print(cntrPixRS.shape)
#    print(uvPixRS.shape)
    for i in np.arange(N):
        pix_ln = np.dot(SQ(smp_vecRS[i,:])-cntrPixRS,uvPixRS)
#        print((pix_ln))
        arcsc_ln[i] = np.sign(pix_ln)*(pix2ang_scale(pix_ln,w).arcsec)
    
    return arcsc_ln

# =============== PLOT_CUT ================
def plot_cut(imgs, mask_ims, noise_ims, hdu_list, inds, cntr, ang, lnlen, axs_img, avgWid):
    # assuming Ang is a list containing two astropy.coordinates.Angle elements:
    # Ang[0]=RA, Ang[1]=Dec
    # ang=[astropy.coordinates.Angle], lnlen=[astropy.coordinates.Angle]
    SQ = np.squeeze
    COS = np.cos
    SIN = np.sin
    
    pltax.Axes(plt.figure('Light Curves'),[0,0,1,1])
    ax = plt.gca()
    
    w = wcs.WCS(hdu_list[0][0].header)
    
    lnlenPix = ang2pix_scale(lnlen, w)
    
    if avgWid<=1:
        Nbtch=1
    else:
        Nbtch=2*np.round(avgWid).astype(int)
    
    Nsmp = np.floor(lnlenPix).astype('int')+1
    smp_mat = np.zeros((inds.size,Nsmp))
    smp_err_mat = np.zeros((inds.size,Nsmp))
    smp_arcsec_mat = np.zeros((inds.size,Nsmp))
    
    deep_smp_mat = np.zeros((inds.size,Nsmp,Nbtch))
    deep_err_mat = np.zeros((inds.size,Nsmp,Nbtch))
    deep_smp_arcsec_mat = np.zeros((inds.size,Nsmp,Nbtch))
    
    angRad = ang.radian;

    [u_th, u_ph] = car_uvec_ang(cntr)
    cntr_car = sky_s2c(cntr)
    uv_car = cntr_car + COS(angRad)*u_ph + SIN(angRad)*u_th

    uv_sphr = sky_c2s(uv_car)

    N = int(deep_smp_mat.size/deep_smp_mat.shape[0])
    for i in np.arange(len(inds)):
        w = wcs.WCS(hdu_list[i][0].header)
        P2W = w.wcs_pix2world
        W2P = w.wcs_world2pix
        cntrPix = W2P(np.array([cntr[0].degree,cntr[1].degree],ndmin=2),0)
        uvPix = W2P(np.array([uv_sphr[0].degree,uv_sphr[1].degree],ndmin=2),0) - cntrPix
        angPixRad = np.arctan2(SQ(uvPix)[1],SQ(uvPix)[0])
        unitvec = np.array([[COS(angPixRad)],[ SIN(angPixRad)]])
        uninorm = np.array([[SIN(angPixRad)],[-COS(angPixRad)]])
#        cntrPix = w.wcs_world2pix(np.array([cntr[0].degree, cntr[1].degree],ndmin=2),0)
        cntrPix = np.reshape(cntrPix,uninorm.shape)
        
        
        
        smp_sum = np.zeros((Nsmp,))
        for n in np.arange(Nbtch):
            cntr_tmp = cntrPix +uninorm*(float(n/2)-float(Nbtch/4))
            str_pnt = cntr_tmp -unitvec*lnlenPix/2
            end_pnt = cntr_tmp +unitvec*lnlenPix/2

            vec = SQ(np.linspace(str_pnt, end_pnt, Nsmp),2)
            print(vec.shape)
            print(str(i) + ', ' + str(n))
            [img_smp, smp_vec] = samp_image(imgs[inds[i]], mask_ims[inds[i]], noise_ims[inds[i]], vec, 'roundloc')
            
            smp_sum = smp_sum + img_smp
            plt.sca(axs_img[inds[i]])
            plt.scatter(smp_vec[:,0],smp_vec[:,1],s=0.1,color='r')
            
            deep_smp_mat[i,:,n] = img_smp
            deep_err_mat[i,:,n], _ = samp_image(noise_ims[inds[i]], mask_ims[inds[i]], noise_ims[inds[i]], vec, 'roundloc')
            deep_smp_arcsec_mat[i,:,n] = vectanpth2arcsec(smp_vec,w,cntrPix,normalize(uvPix))

        try:
            magz = hdu_list[i][0].header['MAGZERO']
            print('used MAGZERO from header: '+str(magz))
        except:
            magz = 30.5
            print('no MAGZERO in header, used default value: '+str(magz))
        fact = np.power(10,-magz/2.5)*1e12
        
        smp_arcsec_mat[i,:] = vectanpth2arcsec(smp_vec,w,cntrPix,normalize(uvPix))
#        print(smp_vec.shape)
        
        smp = smp_sum/Nbtch
        [smp_err, smp_vec] = samp_image(noise_ims[inds[i]], mask_ims[inds[i]], noise_ims[inds[i]], vec, 'roundloc')
#        print(smp_err.shape)
#        smp = smp.copy()*fact
        smp_err = smp_err.copy()*fact
#        smp[smp<0] = np.nan
#        smp_err[smp<0] = np.nan
#        smp=-2.5*np.log10(smp)
#        smp_err=-2.5*np.log10(np.abs((smp-smp_err)/smp))
#        sky_ang = np.arange(img_smp.size)
#        sky_ang = np.linspace(-lnlen.arcsec/2,lnlen.arcsec/2,img_smp.size)
#        plt_sky_ang = smp_arcsec_mat[i,:]
        
        plt_sky_ang = np.reshape(deep_smp_arcsec_mat[i,:,:],(N,))
        plt_smp = np.reshape(deep_smp_mat[i,:,:],(N,))
        plt_smp_err = 0*np.reshape(deep_err_mat[i,:,:],(N,))
        plt.sca(ax)
#        smp[smp<=7]=0
        
        try:
            plt.plot(plt_sky_ang,plt_smp,label=str(inds[i]),linestyle='None',marker='.',markersize=1)
#            plt.errorbar(plt_sky_ang,plt_smp,yerr=plt_smp_err,label=str(inds[i]),capsize=2,ls='None',marker='.',markersize=0.5)
            plt.xlabel('[arcsec]')
        except:
            print('?')
        smp_mat[i,:] = smp
        smp_err_mat[i,:] = smp_err
        
        
    ax.legend()
    return smp_mat, smp_err_mat, smp_arcsec_mat, deep_smp_mat, deep_err_mat, deep_smp_arcsec_mat

# =============== SAMP_IMAGE ================
def samp_image(img, mask, noise, vec, method):
#   img = 2D [yPix*xPix] image np.array
#   vec = 2D [N*2] vector of N sampling locations [yPix, xPix] on img

    if method=='bilinear':
        vc_shp = (vec.shape[0],1)
        fx = np.reshape(np.floor(vec[:,0]),vc_shp)
        fy = np.reshape(np.floor(vec[:,1]),vc_shp)
        cx = np.reshape(np.ceil(vec[:,0]),vc_shp)
        cy = np.reshape(np.ceil(vec[:,1]),vc_shp)
        
        
        x_mod = np.mod(vec[:,0],1)
        y_mod = np.mod(vec[:,1],1)

        vec_fxfy = np.concatenate((fx,fy),1)
        vec_fxcy = np.concatenate((fx,cy),1)
        vec_cxfy = np.concatenate((cx,fy),1)
        vec_cxcy = np.concatenate((cx,cy),1) 
        
        smp_fxfy = smp_ind(img, mask, noise, vec_fxfy)
        smp_fxcy = smp_ind(img, mask, noise, vec_fxcy)
        smp_cxfy = smp_ind(img, mask, noise, vec_cxfy)
        smp_cxcy = smp_ind(img, mask, noise, vec_cxcy)
        
        DP = np.multiply #dot product, elementwise multiplication
        img_smp = DP(DP(smp_fxfy,1-x_mod)+DP(smp_cxfy,x_mod),1-y_mod) + DP(DP(smp_fxcy,1-x_mod)+DP(smp_cxcy,x_mod),y_mod)
        smp_vec = vec

    elif method=='roundloc':
#        vec = np.round(vec)
#        smp_inds = np.ravel_multi_index([vec[:,1].astype(int),vec[:,0].astype(int)], im_shp)
#        img_flat = img.flatten()
#        img_smp = img_flat[smp_inds]
#        axind = int(np.argwhere(np.logical_not(np.array(vec.shape)==2)))
        smp_vec = np.round(vec)
#        smp_vec = np.unique(np.round(vec), axis=axind)
        img_smp = smp_ind(img, mask, noise, smp_vec)

        
    elif method=='interp2d':
        x = np.arange(img.shape[1])
        y = np.arange(img.shape[0])
        f = spi.interp2d(x, y, img, kind='linear')
        
        BigDim = np.argmax(vec.shape)
        Nbd = vec.shape[BigDim]
        if BigDim==0:
#            vecRS = np.reshape(vec, (2,Nbd), order='F')
            vecRS = np.transpose(vec)
        else:
            vecRS = vec
        assert vecRS.shape==(2,Nbd)
        print(vecRS.shape)
        x_s = vecRS[0,:]
        y_s = vecRS[1,:]
        smthng = spi.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], x_s, y_s)
        print(str(type(smthng[0]))+str(len(smthng[0])))
        img_smp = smthng[0]
        print(img_smp.shape)
        smp_vec = vec
        
    return img_smp, smp_vec

# =============== SMP_IND ================
def smp_ind(img, mask, noise, vec):
    
    imgFilt = img_filt(img, mask)

    smps = np.zeros((vec.shape[0],))
    shp = imgFilt.shape
    [vec, notnan] = filt_inds(shp,vec)
    vec = np.round(vec)
    inds = np.ravel_multi_index([vec[:,1].astype(int),vec[:,0].astype(int)], shp)
    img_flat = imgFilt.flatten()
    smps[notnan] = img_flat[inds]
    
#    ArgW = np.argwhere
#    SQ = np.squeeze
    NOT = np.logical_not
    smps[NOT(notnan)] = np.nan
    
    return smps

# =============== IMG_FILT ================
def img_filt(img, mask):
    nullval = np.nan    
#    nullval = 0
    imgFilt = img.astype(float)
    imgFilt[mask>=0x8000] = nullval # kill "bad" pixels
#    imgFilt[(mask>0) & (mask<0x8000)] = nullval # kill "suspicious" pixels
#    imgFilt[mask==0] = nullval # kill "good" pixels
    
    return imgFilt
    
# =============== FILT_INDS ================
def filt_inds(im_shp, vec):
    vec_x = vec[:,0]
    vec_y = vec[:,1]
    
    vec_x[(vec_x<0) | (vec_x>im_shp[1])] = np.NaN
    vec_y[(vec_y<1) | (vec_y>im_shp[0])] = np.NaN
    
    vc_shp = (vec.shape[0],1)
    vec_x = np.reshape(vec_x,vc_shp)
    vec_y = np.reshape(vec_y,vc_shp)
    
    ArgW = np.argwhere
    SQ = np.squeeze
    NOT = np.logical_not
    
    notnan = SQ(NOT(np.isnan(vec_x) | np.isnan(vec_y)))
    inds_notnan = SQ(ArgW(notnan))
    vec_x = vec_x[inds_notnan]
    vec_y = vec_y[inds_notnan]
    
#    vec_x = np.reshape(np.squeeze(vec_x),inds_notnan.shape)
#    vec_y = np.reshape(np.squeeze(vec_y),inds_notnan.shape)
    
    vec = np.concatenate((vec_x,vec_y),1)
    
    return vec, notnan

## =============== LOAD_DIFIMG ================
#__difimg(diff_flnm, msks=None, nerrs=None, home_dir=None):
#    
#    from astropy.io.fits.verify import VerifyWarning
#    import warnings
#    warnings.simplefilter('ignore', category=VerifyWarning)
#
#    NOT = np.logical_not
#    OR  = np.logical_or
#    
#    DFclmn = ['Idate','Tdate','Diff','Mask','Noise']
#    difDF = pd.DataFrame(columns=DFclmn)
#    for ind in np.arange(len(fits_pthfnm)):
#        assert(type(home_dir)==str)
#        if home_dir:
#            if home_dir[-1]=='/':
#                slsh=''
#            else:
#                slsh='/'
#            fits_
#        else:
#            est_home_dir
#            
#        if OR(mask_flnm==False)
##        flnm = home_dir + image_files[ind]
##        image_data.append(fits.getdata(flnm))# + '.fits'))
##        hdu_list.append(fits.open(flnm))# + '.fits'))
#        flnm = home_dir + prefix + image_files[ind] + midfix + tmplt_img + sufix
#        image_data.append(fits.getdata( flnm + '.fits'))
#        mask_img.append(  fits.getdata( flnm + '.mask.fits'))
#        noise_img.append( fits.getdata( flnm + '.noise.fits'))
#        hdu_list.append(fits.open(flnm + '.fits'))
#        
##        mat = noise_img[ind]
#        try:
#            magz = hdu_list[ind][0].header['MAGZERO']
#        except:
#            magz = 30.5
#        fact[ind] = np.power(10,-magz/2.5)*1e12
#        
#        mat = image_data[ind]
##        mat = img_filt(image_data[ind],  mask_img[ind])
##        mat[np.abs(mat)<2*np.abs(noise_img[ind])]=np.nan
#        mat = fact[ind]*mat
##        if gss_FWHM:
##            mat = gaussian_filter(mat, sigma=gss_FWHM/(2*np.sqrt(2*np.log(2))))
#
#        image_data[ind] = mat
#        
#        w = wcs.WCS(hdu_list[ind][0].header)
#        wcs_list.append(w)
#        
#        fig = plt.figure(ind+1)
#        fig.canvas.mpl_connect('key_press_event', press)
#        mng = plt.get_current_fig_manager()
#        mng.window.showMaximized()
#        if ind==0:
##            ax.append(fig.add_subplot(212))
##            ax_img.append(fig.add_subplot(211,projection=w))
#            ax_img.append(fig.add_subplot(111,projection=w))
#        else:
##            ax.append(fig.add_subplot(212,sharex=ax[ind-1],sharey=ax[ind-1]))
##            ax_img.append(fig.add_subplot(211,sharex=ax_img[ind-1],sharey=ax_img[ind-1],projection=w))
#            ax_img.append(fig.add_subplot(111,sharex=ax_img[ind-1],sharey=ax_img[ind-1],projection=w))
#            
#        clim = 120
#        plt.sca(ax_img[ind])
#        ax_img[ind].set_facecolor((0.5*135/255,0.5*206/255,0.5*235/255))#'k')
##        ax_img[ind].title.set_text(sdate(image_files[ind]))
##        plt.subplot(projection=w)
#        plt.imshow(mat, vmin=-clim, vmax=clim, origin='lower', cmap='gray', interpolation='none')#, extent=ext, aspect=asp)
#        plt.grid(color='white', ls='solid')
##        plt.plot(np.array([SEwrld[0][0], NWwrld[0][0]]),np.array([SEwrld[0][1], NWwrld[0][1]]),color='r',transform=ax_img[ind].get_transform('world'))
##        plt.scatter(SEwrld[0][0], SEwrld[0][1], s=20)
#        
##        plt.imshow(mat, cmap='gray', vmin=-clim, vmax=clim)
#        
##        (x,y)=np.unique(mat,return_counts=True)
##        
##        plt.sca(ax[ind])    
##        plt.scatter(x,np.log10(y),2)
#        print('loaded file no. '+str(ind))
#    
#    return image_data, mask_img, noise_img, hdu_list, ax_img, wcs_list

# =============== PLOT_DIF ================
def plot_dif(base_img,diff_imgs):
    image_data = list()
    ax_img=list()
    
    for ind in np.arange(len(diff_imgs)):
        
        fig = plt.figure(ind+100)
#        mng = plt.get_current_fig_manager()
#        mng.window.showMaximized()
        if ind==0:
            ax_img.append(fig.add_subplot(111))
        else:
            ax_img.append(fig.add_subplot(111,sharex=ax_img[ind-1],sharey=ax_img[ind-1]))
            
        clim = 70
        
        image_data.append(diff_imgs[ind]-base_img)
        plt.sca(ax_img[ind])
        plt.imshow(image_data[ind], cmap='gray', vmin=-clim, vmax=clim)
        
#        (x,y)=np.unique(mat,return_counts=True)
#        
#        plt.sca(ax[ind])    
#        plt.scatter(x,np.log10(y),2)
    
    return image_data, ax_img


# =============== FLNM2TIME ================
def flnm2time(names):
    # names = list of str, each with first 8 chars representing date of later epoch in diff image
    # times = int representing number of days since names[0]
    times = np.zeros(len(names)).astype(float)
#    [y0,m0,d0] = sdate(names[0])
    for i in np.arange(len(names)):
#        [y,m,d] = sdate(names[i])
#        times[i] = 365*(y-y0) + 30*(m-m0) + (d-d0)
        d_str = names[i][0:4] + '-' + names[i][4:6] + '-' + names[i][6:8] + 'T00:00:00'
        times[i] = Time(d_str, format='isot', scale='utc').mjd
    return times

# =============== SDATE ================
def sdate(str_date):
    year =  int(str_date[0:4])
    month = int(str_date[4:6])
    day =   int(str_date[6:8])
    
    return year, month, day

# ==============ARCLT=================
def plot_arclt(LE,SN,arln,ax,wcs):
    SQ = np.squeeze
    XP = np.cross
    DP = np.dot
    
    LEcar = sky_s2c(LE)
    SNcar = sky_s2c(SN)
    mid_car = (LEcar+SNcar)/2
    mid_sph = sky_c2s(mid_car)

    Perp = XP(SNcar,LEcar)
    nPerp = normalize(Perp)
    [u_th, u_ph] = car_uvec_ang(LE)
    arcltPA = Angle(np.mod(np.arctan2(DP(SQ(u_ph),SQ(nPerp)),DP(SQ(u_th),SQ(nPerp))),2*np.pi),u.radian)
    angSeper = Angle(np.arcsin(v_norma(Perp)),u.radian)

    E1 = sky_c2s(LEcar+nPerp*arln.radian/2)
    E2 = sky_c2s(LEcar-nPerp*arln.radian/2)
    Ra = np.array([E1[0].degree, E2[0].degree])
    Dec = np.array([E1[1].degree, E2[1].degree])
    
    plt.sca(ax)
    plt.plot(Ra, Dec, color='b', transform = ax.get_transform('world'))
    
    plt.plot(np.array([SN[0].degree, LE[0].degree]), np.array([SN[1].degree, LE[1].degree]), color='c', transform = ax.get_transform('world'))
    
    plt.scatter(LE[0].degree, LE[1].degree, s=20,color='m', transform = ax.get_transform('world'))
    plt.scatter(SN[0].degree, SN[1].degree, s=20,color='g', transform = ax.get_transform('world'))
    plt.scatter(mid_sph[0].degree, mid_sph[1].degree, s=20,color='y', transform = ax.get_transform('world'))
    return arcltPA, angSeper, mid_sph

# §§§ VECTOR MATH §§§
def normalize(vec):
    norma = v_norma(vec)
    n_vec = vec/norma
    return n_vec

def v_norma(vec):
    SQ = np.squeeze
    norma = np.sqrt(np.dot(SQ(vec),SQ(vec)))
    return norma

def car_uvec_ang(Ang):
    phi = Ang[0].radian
    th  = Ang[1].radian
    
    COS = np.cos
    SIN = np.sin
    
    u_th = np.array([-SIN(th)*COS(phi),-SIN(th)*SIN(phi),COS(th)],ndmin=2)
    u_ph = np.array([-SIN(phi),COS(phi),0],ndmin=2)
    
    return u_th, u_ph

def sky_s2c(Ang):
    # assuming Ang is a list containing two astropy.coordinates.Angle elements:
    # Ang[0]=RA, Ang[1]=Dec
    a = Ang[0].radian
    d = Ang[1].radian
    
    x = np.cos(d)*np.cos(a)
    y = np.cos(d)*np.sin(a)
    z = np.sin(d)
    vec = np.array([x,y,z],ndmin=2)
    return vec

def sky_c2s(vec):
    SQ = np.squeeze
    vec = SQ(normalize(vec))
    Ang = list()
    Ang.append(Angle(np.mod(np.arctan2(vec[1],vec[0]),2*np.pi),u.radian))
    Ang.append(Angle(np.arcsin(vec[2]),u.radian))
    return Ang


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= OLD  VERSIONS =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =============== LOAD_DIFIMG ================
def load_difimg(home_dir, prefix, midfix, sufix, tmplt_img, image_files, gss_FWHM=None):
    from astropy.io.fits.verify import VerifyWarning
    import warnings
    warnings.simplefilter('ignore', category=VerifyWarning)
    image_data = list()
    mask_img = list()
    noise_img = list()
    hdu_list = list()
    global wcs_list
    wcs_list = list()
    fact =np.ones((len(image_files),),dtype=float)# [ 12/7, 12/9, 12/7, 12/7, 1, 120/84,12/4,12/6, 1] #
    ax=list()
    ax_img=list()
    print('\nFUCK!\n')
#    plt.close('all')
    
    for ind in np.arange(len(image_files)):
#        flnm = home_dir + image_files[ind]
#        image_data.append(fits.getdata(flnm))# + '.fits'))
#        hdu_list.append(fits.open(flnm))# + '.fits'))
        flnm = home_dir + prefix + image_files[ind] + midfix + tmplt_img + sufix
        image_data.append(fits.getdata( flnm + '.fits'))
        mask_img.append(  fits.getdata( flnm + '.mask.fits'))
        noise_img.append( fits.getdata( flnm + '.noise.fits'))
#        mask_img.append(np.zeros(image_data[ind].shape))
#        noise_img.append(np.zeros(image_data[ind].shape))
        hdu_list.append(fits.open(flnm + '.fits'))
        
#        mat = noise_img[ind]
        try:
            dcmp_hdu = fits.open(flnm+'.dcmp')
            magz = dcmp_hdu[0].header['ZPTMAG00']#hdu_list[ind][0].header['MAGZERO']
            fwhm = dcmp_hdu[0].header['FWHM']
        except:
            try:
                Iflnm = home_dir + prefix + image_files[ind] + '_stch_2.sw'
                Tflnm = home_dir + prefix + tmplt_img + '_stch_2.sw'
                Idcmp_hdu = fits.open(Iflnm+'.dcmp')
                Tdcmp_hdu = fits.open(Tflnm+'.dcmp')
                magz = Idcmp_hdu[0].header['ZPTMAG']#hdu_list[ind][0].header['MAGZERO']
                fwhm = np.max([Idcmp_hdu[0].header['FWHM'],Tdcmp_hdu[0].header['FWHM']])
            except:
                print('?!')
                magz = 30
                fwhm = 1
        fact[ind] = np.power(10,-magz/2.5)*1e12
        print(ind)
        print('FWHM: '+str(fwhm))
        print('ZPTMAG: '+str(magz))
        
        mat = image_data[ind]
#        mat = img_filt(mat,  mask_img[ind])
#        mat[np.abs(mat)<2*np.abs(noise_img[ind])]=np.nan
        mat = fact[ind]*mat
#        gss_FWHM = np.sqrt((10**2)-(fwhm**2))#9.117559
        if gss_FWHM:
            mat = gaussian_filter(mat, sigma=gss_FWHM/(2*np.sqrt(2*np.log(2))))

        image_data[ind] = mat
        
        w = wcs.WCS(hdu_list[ind][0].header)
#        w = manual_wcs(hdu_list[ind][0].header)
        wcs_list.append(w)
        
        fig = plt.figure()#ind+1)
        fig.canvas.mpl_connect('key_press_event', press)
        global first_click
        first_click = True
        fig.canvas.mpl_connect('button_press_event', onpress)
        fig.canvas.mpl_connect('button_release_event', onrelease)
        fig.canvas.mpl_connect('motion_notify_event', onmove)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        if ind==0:
#            ax.append(fig.add_subplot(212))
#            ax_img.append(fig.add_subplot(211,projection=w))
            ax_img.append(fig.add_subplot(111,projection=w))
        else:
#            ax.append(fig.add_subplot(212,sharex=ax[ind-1],sharey=ax[ind-1]))
#            ax_img.append(fig.add_subplot(211,sharex=ax_img[ind-1],sharey=ax_img[ind-1],projection=w))
            ax_img.append(fig.add_subplot(111,sharex=ax_img[0],sharey=ax_img[0],projection=w))
            
        clim = 120
        plt.sca(ax_img[ind])
        ax_img[ind].set_facecolor((0.5*135/255,0.5*206/255,0.5*235/255))#'k')
        ax_img[ind].title.set_text(image_files[ind])#sdate(image_files[ind]))
#        plt.subplot(projection=w)
        plt.imshow(mat, vmin=-clim, vmax=clim, origin='lower', cmap='gray', interpolation='none')#, extent=ext, aspect=asp)
        plt.grid(color='white', ls='solid')
#        plt.plot(np.array([SEwrld[0][0], NWwrld[0][0]]),np.array([SEwrld[0][1], NWwrld[0][1]]),color='r',transform=ax_img[ind].get_transform('world'))
#        plt.scatter(SEwrld[0][0], SEwrld[0][1], s=20)
        
#        plt.imshow(mat, cmap='gray', vmin=-clim, vmax=clim)
        
#        (x,y)=np.unique(mat,return_counts=True)
#        
#        plt.sca(ax[ind])    
#        plt.scatter(x,np.log10(y),2)
        print('loaded file no. '+str(ind))
    
    return image_data, mask_img, noise_img, hdu_list, ax_img, wcs_list

def load_difimg_atlas(home_dir, image_files, gss_FWHM=None, FS=False):
#    from astropy.io.fits.verify import VerifyWarning
#    import warnings
#    warnings.simplefilter('ignore', category=VerifyWarning)
    Zscale = ZScaleInterval()
    image_data = list()
    mask_img = list()
    noise_img = list()
    hdu_list = list()
    dcmp_hdus= list()
    noise_hdus= list()
    mask_hdus= list()
    clicks = list()
    global wcs_list
    wcs_list = list()
    fact =np.ones((len(image_files),),dtype=float)# [ 12/7, 12/9, 12/7, 12/7, 1, 120/84,12/4,12/6, 1] #
    ax=list()
    ax_img=list()
    print('\nFUCK!\n')
#    plt.close('all')
    
    for ind in np.arange(len(image_files)):
        flnm = home_dir + image_files[ind]
        print('File no. '+str(ind)+flnm)
#        image_data.append(fits.getdata(flnm))# + '.fits'))
#        hdu_list.append(fits.open(flnm))# + '.fits'))
#        flnm = home_dir + prefix + image_files[ind] + midfix + tmplt_img + sufix
        print(flnm)
#        image_data.append(fits.getdata( flnm))# + '.fits'))
#        mask_hdus.append(  fits.open( flnm + '.mask.fits'))
#        mask_img.append(   mask_hdus[ind][0].data)
#        noise_hdus.append( fits.open( flnm + '.noise.fits'))
#        noise_img.append(  noise_hdus[ind][0].data)
#        mask_img.append(np.zeros(image_data[ind].shape))
#        noise_img.append(np.zeros(image_data[ind].shape))
        hdu_list.append(fits.open(flnm+ '.fits'))
        image_data.append(hdu_list[ind][0].data)
        
#        mat = noise_img[ind]
        try:
            dcmp_hdus.append(  fits.open( flnm + '.dcmp'))
            if image_files[ind][-4:]=='diff':
                magz = dcmp_hdus[ind][0].header['ZPTMAG00']#hdu_list[ind][0].header['MAGZERO']
            else:
                magz = dcmp_hdus[ind][0].header['ZPTMAG']
            fwhm = dcmp_hdus[ind][0].header['FWHM']
            skysig=hdu_list[ind][0].header['SKYADU']
        except:
#            try:
##                Iflnm = home_dir + prefix + image_files[ind] + '_stch_2.sw'
##                Tflnm = home_dir + prefix + tmplt_img + '_stch_2.sw'
##                Idcmp_hdu = fits.open(Iflnm+'.dcmp')
##                Tdcmp_hdu = fits.open(Tflnm+'.dcmp')
#                magz = Idcmp_hdu[0].header['ZPTMAG']#hdu_list[ind][0].header['MAGZERO']
#                fwhm = np.max([Idcmp_hdu[0].header['FWHM'],Tdcmp_hdu[0].header['FWHM']])
#            except:
            print('?!')
            dcmp_hdus.append('didn''t find dcmp file...')
            magz = 30
            fwhm = 1
            skysig = 9587
        fact[ind] = np.power(10,-magz/2.5)*1e12
        print(fact[ind])
        print(ind)
        print('FWHM: '+str(fwhm))
        print('ZPTMAG: '+str(magz))
        print('SKYSIG: '+str(skysig))
        
        mat = image_data[ind]*1.0
#        gss_FWHM = 4#np.sqrt((10**2)-(fwhm**2))#9.117559
        if gss_FWHM:
            mat = gaussian_filter(mat, sigma=gss_FWHM/(2*np.sqrt(2*np.log(2))))
#        mat = img_filt(mat,  mask_img[ind])
#        mat[np.abs(mat)<2*np.abs(noise_img[ind])]=np.nan
#        mat[np.abs(mat)>100*np.abs(noise_img[ind])]=np.nan
#        mat[mat<0]=np.nan
#        mat[np.abs(mat)<2*np.abs(skysig)]=np.nan
#        mat[np.abs(mat)<2.7*np.abs(skysig)]=np.nan
#        mat[np.abs(mat)>6*np.abs(skysig)]=np.nan
#        mat = fact[ind]*mat
#        mat[np.abs(mat)>650]=np.nan
#        mat[np.abs(mat)<400]=np.nan
#        mat[np.abs(mat)==0]=np.nan

        image_data[ind] = mat
        
#        w = wcs.WCS(hdu_list[ind][0].header)
        w = manual_wcs(hdu_list[ind][0].header)
        wcs_list.append(w)
        
        fig = plt.figure()#ind+1)
        fig.canvas.mpl_connect('key_press_event', press)
        global first_click, press_bool, move
        first_click = True
        press_bool=False
        move = False
        fig.canvas.mpl_connect('button_press_event', onpress)
        fig.canvas.mpl_connect('button_release_event', onrelease)
        fig.canvas.mpl_connect('motion_notify_event', onmove)
        if FS:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
        if ind==0:
#            ax.append(fig.add_subplot(212))
#            ax_img.append(fig.add_subplot(211,projection=w))
            ax_img.append(fig.add_subplot(111,projection=w))
        else:
#            ax.append(fig.add_subplot(212,sharex=ax[ind-1],sharey=ax[ind-1]))
#            ax_img.append(fig.add_subplot(211,sharex=ax_img[ind-1],sharey=ax_img[ind-1],projection=w))
            ax_img.append(fig.add_subplot(111,sharex=ax_img[0],sharey=ax_img[0],projection=w))
#        clicks.append(Click(ax_img[ind], w))
#        mat = np.log10(mat)
        clim = Zscale.get_limits(mat)#1e3#np.nanmax(mat)
        print(clim)
#        mat=clim-mat
#        print(clim)
#        vvv=np.unravel_index(np.nanargmax(mat),mat.shape)
#        print(vvv)
#        vvvv=w.wcs_pix2world(np.array([vvv[1], vvv[0]],ndmin=2),0)
#        print(vvvv)
#        print(Angle(vvvv[0][0],'deg').hms)
#        print(Angle(vvvv[0][1],'deg').dms)
#        print(mat.flatten()[np.nanargmax(mat)])
#        print(np.nanmin(mat))
#        print('count= '+str(np.sum(~np.isnan(mat).flatten())))
        plt.sca(ax_img[ind])
        ax_img[ind].set_facecolor((0,0,0))#'k')
#        ax_img[ind].set_facecolor((0.5*135/255,0.5*206/255,0.5*235/255))#'k')
        ax_img[ind].title.set_text(image_files[ind])#sdate(image_files[ind]))
#        plt.subplot(projection=w)
#        plt.imshow(mat, vmin=500, vmax=2e3, origin='lower', cmap='gray', interpolation='none')#, extent=ext, aspect=asp)
        plt.imshow(mat, vmin=-clim[1], vmax=clim[1], origin='lower', cmap='gray', interpolation='none')#, extent=ext, aspect=asp)
        plt.grid(color='white', ls='solid')
#        plt.plot(np.array([SEwrld[0][0], NWwrld[0][0]]),np.array([SEwrld[0][1], NWwrld[0][1]]),color='r',transform=ax_img[ind].get_transform('world'))
#        plt.scatter(SEwrld[0][0], SEwrld[0][1], s=20)
        
#        plt.imshow(mat, cmap='gray', vmin=-clim, vmax=clim)
        
#        (x,y)=np.unique(mat,return_counts=True)
#        
#        plt.sca(ax[ind])    
#        plt.scatter(x,np.log10(y),2)
        print('loaded file no. '+str(ind))
        hdu_list_list = [hdu_list, dcmp_hdus, noise_hdus, mask_hdus]
    
    return image_data, hdu_list_list, ax_img, wcs_list


def load_difimg_atlas_HDU_LOADED(hdu_list_list, image_files, gss_FWHM=None, FS=False):
#    from astropy.io.fits.verify import VerifyWarning
#    import warnings
#    warnings.simplefilter('ignore', category=VerifyWarning)
    Zscale = ZScaleInterval()
    image_data = list()
    mask_img = list()
    noise_img = list()
    hdu_list =   list(hdu_list_list[0])
    dcmp_hdus =  list(hdu_list_list[1])
    noise_hdus = list(hdu_list_list[2])
    mask_hdus =  list(hdu_list_list[3])
#    clicks = list()
    global wcs_list
    wcs_list = list()
    fact =np.ones((len(image_files),),dtype=float)# [ 12/7, 12/9, 12/7, 12/7, 1, 120/84,12/4,12/6, 1] #
#    ax=list()
    ax_img=list()
    print('\nFUCK!\n')
#    plt.close('all')
    
    for ind in np.arange(len(hdu_list)):

#        mask_img.append(   mask_hdus[ind][0].data)
#        noise_img.append(  noise_hdus[ind][0].data)
        image_data.append(hdu_list[ind][0].data)
        
        try:
            if image_files[ind][-4:]=='diff':
                magz = dcmp_hdus[ind][0].header['ZPTMAG00']#hdu_list[ind][0].header['MAGZERO']
            else:
                magz = dcmp_hdus[ind][0].header['ZPTMAG']
            fwhm = dcmp_hdus[ind][0].header['FWHM']
            skysig=hdu_list[ind][0].header['SKYADU']
        except:
#            try:
##                Iflnm = home_dir + prefix + image_files[ind] + '_stch_2.sw'
##                Tflnm = home_dir + prefix + tmplt_img + '_stch_2.sw'
##                Idcmp_hdu = fits.open(Iflnm+'.dcmp')
##                Tdcmp_hdu = fits.open(Tflnm+'.dcmp')
#                magz = Idcmp_hdu[0].header['ZPTMAG']#hdu_list[ind][0].header['MAGZERO']
#                fwhm = np.max([Idcmp_hdu[0].header['FWHM'],Tdcmp_hdu[0].header['FWHM']])
#            except:
            print('?!')
            magz = 30
            fwhm = 1
            skysig = 9587
        fact[ind] = np.power(10,-magz/2.5)*1e12
        print(fact[ind])
        print(ind)
        print('FWHM: '+str(fwhm))
        print('ZPTMAG: '+str(magz))
        print('SKYSIG: '+str(skysig))
        
        mat = hdu_list[ind][0].data*1.0
        mask_mat = mask_hdus[ind][0].data*1.0
        noise_mat = noise_hdus[ind][0].data*1.0
#        gss_FWHM = 4#np.sqrt((10**2)-(fwhm**2))#9.117559
        if gss_FWHM:
            mat = gaussian_filter(mat, sigma=gss_FWHM/(2*np.sqrt(2*np.log(2))))
#        mat[:,0:3000] = np.nan
#        mat[0:1100,:] = np.nan
#        mat = img_filt(mat,  mask_mat)
##        mat[np.abs(mat)<1.5*np.abs(noise_mat)]=np.nan
##        mat[np.abs(mat)>110]=np.nan
#        mat[mat<0]=np.nan
#        mat[np.abs(mat)<2*np.abs(skysig)]=np.nan
#        mat[np.abs(mat)<2.7*np.abs(skysig)]=np.nan
#        mat[np.abs(mat)>6*np.abs(skysig)]=np.nan
#        mat = fact[ind]*mat
#        mat[np.abs(mat)>650]=np.nan
#        mat[np.abs(mat)<400]=np.nan
#        mat[np.abs(mat)==0]=np.nan

        
        
        w = wcs.WCS(hdu_list[ind][0].header)
#        w = manual_wcs(hdu_list[ind][0].header)
        wcs_list.append(w)
        
        fig = plt.figure()#ind+1)
        fig.canvas.mpl_connect('key_press_event', press)
        global first_click, press_bool, move
        first_click = True
        press_bool=False
        move = False
        fig.canvas.mpl_connect('button_press_event', onpress)
        fig.canvas.mpl_connect('button_release_event', onrelease)
        fig.canvas.mpl_connect('motion_notify_event', onmove)
        if FS:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
        if ind==0:
#            ax.append(fig.add_subplot(212))
#            ax_img.append(fig.add_subplot(211,projection=w))
            ax_img.append(fig.add_subplot(111,projection=w))
        else:
#            ax.append(fig.add_subplot(212,sharex=ax[ind-1],sharey=ax[ind-1]))
#            ax_img.append(fig.add_subplot(211,sharex=ax_img[ind-1],sharey=ax_img[ind-1],projection=w))
            ax_img.append(fig.add_subplot(111,sharex=ax_img[0],sharey=ax_img[0],projection=w))
#        clicks.append(Click(ax_img[ind], w))
#        mat = np.log10(mat)
        clim = Zscale.get_limits(mat)#1e3#np.nanmax(mat)
        print(clim)
#        mat[mat<(0.95*clim[1])] = np.nan
#        
#        while np.sum(~np.isnan(mat))>500:
#            mat = gaussian_filter(mat, sigma=2/(2*np.sqrt(2*np.log(2))))
#        mat=clim-mat
#        print(clim)
#        vvv=np.unravel_index(np.nanargmax(mat),mat.shape)
#        print(vvv)
#        vvvv=w.wcs_pix2world(np.array([vvv[1], vvv[0]],ndmin=2),0)
#        print(vvvv)
#        print(Angle(vvvv[0][0],'deg').hms)
#        print(Angle(vvvv[0][1],'deg').dms)
#        print(mat.flatten()[np.nanargmax(mat)])
#        print(np.nanmin(mat))
#        print('count= '+str(np.sum(~np.isnan(mat).flatten())))
        plt.sca(ax_img[ind])
#        ax_img[ind].set_facecolor((0.5,0.5,0.5))#'k')
        ax_img[ind].set_facecolor((0.5*135/255,0.5*206/255,0.5*235/255))#'k')
        ax_img[ind].title.set_text(image_files[ind])#sdate(image_files[ind]))
#        plt.subplot(projection=w)
#        plt.imshow(mat, vmin=500, vmax=2e3, origin='lower', cmap='gray', interpolation='none')#, extent=ext, aspect=asp)
        plt.imshow(mat, vmin=0.95*clim[0], vmax=clim[1], origin='lower', cmap='gray', interpolation='none')#, extent=ext, aspect=asp)
        plt.grid(color='white', ls='solid')
#        plt.plot(np.array([SEwrld[0][0], NWwrld[0][0]]),np.array([SEwrld[0][1], NWwrld[0][1]]),color='r',transform=ax_img[ind].get_transform('world'))
#        plt.scatter(SEwrld[0][0], SEwrld[0][1], s=20)
        
#        plt.imshow(mat, cmap='gray', vmin=-clim, vmax=clim)
        
#        (x,y)=np.unique(mat,return_counts=True)
#        
#        plt.sca(ax[ind])    
#        plt.scatter(x,np.log10(y),2)
        print('loaded file no. '+str(ind))
        image_data[ind] = mat
    return image_data, ax_img, wcs_list
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
def manual_wcs(header):
    w = wcs.WCS()
    w.wcs.ctype=['LINEAR','LINEAR']#[header['CTYPE1'], header['CTYPE2']]#['RA---TAN','DEC--TAN']#
    w.pixel_shape=[header['NAXIS1'],header['NAXIS2']]
    w.wcs.cd=np.array([[header['CD1_1'], header['CD1_2']],[header['CD2_1'], header['CD2_2']]])
    w.wcs.crpix=[header['CRPIX1'], header['CRPIX2']]
    w.wcs.crval=[header['CRVAL1'], header['CRVAL2']]
    w.wcs.cunit=[header['CUNIT1'], header['CUNIT2']]
    return w




def onclick(event):
    ix, iy = event.xdata, event.ydata
    w_ind = event.inaxes.figure.number -1#plt.gcf().number -1
    w=wcs_list[w_ind]
    coord_tmp = Angle(w.wcs_pix2world(np.array([ix,iy],ndmin=2),0),u.deg)[0]
    global first_click
    global first_ax
#    print('FC: '+str(first_click))
    if first_click:
        first_ax = event.inaxes
        print('\n\n'+event.inaxes.title.get_text())
        print('==================================\nPixel_1: x = %d, y = %d'%(ix, iy))
        
        print('World_1: RA = %.8f [deg], DEC = %.8f [deg]'%(coord_tmp[0].deg,coord_tmp[1].deg))
        plt.sca(first_ax)
        plt.text(ix,iy,'(%.2f,%.2f) [deg,deg]'%(coord_tmp[0].deg,coord_tmp[1].deg), c='k')
        global coord1
        # coord1 = coord_tmp  
        coord1 = SkyCoord(coord_tmp[0],coord_tmp[1],frame='fk5')
        first_click = False
    elif event.inaxes==first_ax:
        print('Pixel_2: x = %d, y = %d'%(ix, iy))
        print('World_2: RA = %.8f [deg], DEC = %.8f [deg]'%(coord_tmp[0].deg,coord_tmp[1].deg))
        first_click = True
        # coord2 = coord_tmp
        coord2 = SkyCoord(coord_tmp[0],coord_tmp[1],frame='fk5')
        coord_tmp1 = Angle([coord1.ra.deg,coord1.dec.deg],u.deg)
        coord_tmp2 = Angle([coord2.ra.deg,coord2.dec.deg],u.deg)
        [arcltPA, angSeper, mid_sph] = plot_arclt(coord_tmp2,coord_tmp1,Angle('0d1m0s'),event.inaxes,w)
        arcltPA = (coord2.position_angle(coord1) + Angle(180,u.deg)).wrap_at(360 * u.deg)
        angSeper = coord2.separation(coord1)
        mid_sph = coord1.directional_offset_by(arcltPA,angSeper/2)
        plt.text(ix,iy,'PA: %.1f [deg]'%(arcltPA.deg), c='r')#, bbox=dict(fill=False, edgecolor='red', linewidth=2))
        print('World_mid: RA = %.8f [deg], DEC = %.8f [deg]'%(mid_sph.ra.deg,mid_sph.dec.deg))
        print('PA: %.1f [deg]'%(arcltPA.deg))
        print('Ang. sep.: '+str(angSeper.arcsec)+' [arcsec]')
        print('==================================\n\n')
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
#class Click():
#    def __init__(self, ax, w):
#        print('INIT')
#        self.w=w
#        self.ax=ax
#        self.press=False
#        self.move = False
#        self.first_click = True
#        self.ax.figure.canvas.mpl_connect('button_press_event', self.onpress)
#        self.ax.figure.canvas.mpl_connect('button_release_event', self.onrelease)
#        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)
#        return
#
#    def onclick(self,event):
#        if event.inaxes == self.ax:
#            ix, iy = event.xdata, event.ydata
#            print('HHHH@@@@@')
##                w_ind = event.inaxes.figure.number -1
##                self.w=wcs_list[w_ind]
#            
#            coord_tmp = Angle(self.w.wcs_pix2world(np.array([ix,iy],ndmin=2),0),u.deg)[0]
#            
#            print('FC: '+str(self.first_click))
#            if self.first_click:
#                print('\n\n==================================\nPixel_1: x = %d, y = %d'%(ix, iy))
#                print('World_1: RA = %.8f [deg], DEC = %.8f [deg]'%(coord_tmp[0].deg,coord_tmp[1].deg))
#                self.coord1 = coord_tmp    
#                self.first_click = False
#            else:
#                print('Pixel_2: x = %d, y = %d'%(ix, iy))
#                print('World_2: RA = %.8f [deg], DEC = %.8f [deg]'%(coord_tmp[0].deg,coord_tmp[1].deg))
#                self.first_click = True
#                self.coord2 = coord_tmp
#                [arcltPA, angSeper] = plot_arclt(self.coord2,self.coord1,Angle('0d1m0s'),event.inaxes,self.w)
#                print('PA: '+str(arcltPA.deg)+' [deg]')
#                print('Ang. sep.: '+str(angSeper.arcmin)+' [arcmin]')
#                print('==================================\n\n')
#            return
#    def onpress(self,event):
#        self.press=True
#        return
#    def onmove(self,event):
#        if self.press:
#            self.move=True
#        return
#    def onrelease(self,event):
#        if self.press and not self.move:
#            self.onclick(event)
#        self.press=False; self.move=False
#        return


#fig, (ax1, ax2) = plt.subplots(1, 2)
## Plot some random scatter data
#ax2.scatter(np.random.uniform(0., 10., 10), np.random.uniform(0., 10., 10))
#click = Click(ax2, func, button=1)
#plt.show()
#
#def onclick(event):
##    global ix, iy
#    ix, iy = event.xdata, event.ydata
#    w_ind = plt.gcf().number -1
#    w=wcs_list[w_ind]
#    
#    coord_tmp = Angle(w.wcs_pix2world(np.array([ix,iy],ndmin=2),0),u.deg)[0]
#    
#    global first_click
#    print('FC: '+str(first_click))
#    if first_click:
#        print('\n\n==================================\nPixel_1: x = %d, y = %d'%(ix, iy))
#        print('World_1: RA = %.8f [deg], DEC = %.8f [deg]'%(coord_tmp[0].deg,coord_tmp[1].deg))
#        global coord1
#        coord1 = coord_tmp    
#        first_click = False
#    else:
#        print('Pixel_2: x = %d, y = %d'%(ix, iy))
#        print('World_2: RA = %.8f [deg], DEC = %.8f [deg]'%(coord_tmp[0].deg,coord_tmp[1].deg))
#        first_click = True
#        coord2 = coord_tmp
#        [arcltPA, angSeper] = plot_arclt(coord2,coord1,Angle('0d1m0s'),plt.gca(),w)
#        print('PA: '+str(arcltPA.deg)+' [deg]')
#        print('Ang. sep.: '+str(angSeper.arcmin)+' [arcmin]')
#        print('==================================\n\n')