#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:06:02 2019

@author: roeepartoush
"""

import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.axes as pltax

import scipy.interpolate as spi
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from scipy import ndimage
from scipy import stats

import pandas as pd

from astropy.io import fits
from astropy import wcs
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import Angle, SphericalRepresentation, CartesianRepresentation, UnitSphericalRepresentation
from astropy import wcs
from astropy.coordinates import SkyCoord

import LEplots

def getAavgDiff(DIFF_df):
    images = []
    ref_FWHM = Angle(list(DIFF_df['FWHM_ang'])).max()
    for i in np.arange(len(DIFF_df)):
        HDU = DIFF_df.iloc[i]['Diff_HDU']
        try:
            mask = DIFF_df.iloc[i]['Mask_HDU'].data
        except Exception as e:
            print('\n***ERROR:*** '+str(e))
            mask = np.zeros(HDU.data.shape)
        image = matchFWHM(DIFF_df,i,ref_FWHM)*1.0#/float(HDU.header['KSUM00'])
        image[~mask2bool(mask)] = np.nan
        images.append(image)
    return np.mean(np.stack(images,axis=0), axis=0)

def matchFWHM(DIFF_df,ind,ref_FWHM):
    image = DIFF_df.iloc[ind]['Diff_HDU'].data
    fwhm = DIFF_df.iloc[ind]['FWHM'] # [pixels]
    header = DIFF_df.iloc[ind]['Diff_HDU'].header
    w = wcs.WCS(header)
    gss_FWHM = np.sqrt(((ang2pix(w,ref_FWHM))**2)-(fwhm**2))
    image = gaussian_filter(image, sigma=gss_FWHM/(2*np.sqrt(2*np.log(2))))
    return image

def getFluxProfile(DIFF_df, slitFPdf, width=None, PSFeq_overFWHM=None, N_bins=30, REF_image=None, bin_OL = 0, uniform_wcs=False):
    FP_df_lst=[]
    maxFWHM = Angle(list(DIFF_df['FWHM_ang'])).max()
    print('maxFWHM='+str(maxFWHM.deg))
    images=[]
#    inds_mask_lst=[]
    inds_intersect_lst=[]
    for indS in tqdm(np.arange(len(slitFPdf))):
        print(indS)
        FPcntr_world = scalarize(slitFPdf.iloc[indS]['Orig'])
        FP_pa_world  = scalarize(slitFPdf.iloc[indS]['PA']       )
        FPlen_world  = scalarize(slitFPdf.iloc[indS]['Length']   )
        width =        scalarize(slitFPdf.iloc[indS]['WIDTH']       )
        
#        if indS>0:
#            del FP_df
        FP_df = pd.DataFrame(index=DIFF_df.index, columns=['WrldVec','PixVec','ProjAng','PerpAng','WrldWidth','WrldCorners','WrldCntrs',
                                                           'FluxProfile_ADU','FluxProfile_ADUcal','FluxProfile_MAG',
                                                           'NoiseProfile_ADU','ZPTMAG',
                                                           'FP_Ac_bin_x', 'FP_Ac_bin_y', 'FP_Ac_bin_yerr',
                                                           'FP_Ac_bin_yBref', 'FP_Ac_bin_ystdBref','FP_Ac_bin_Cnt'])
        
        for indD in tqdm(np.arange(len(DIFF_df))):
#            image = DIFF_df.iloc[indD]['Diff_HDU'].data
            HDU = DIFF_df.iloc[indD]['Diff_HDU']
            header = HDU.header#DIFF_df.iloc[indD]['Diff_HDU'].header
            # mask = DIFF_df.iloc[indD]['Mask_mat']
            # noise = DIFF_df.iloc[indD]['Noise_mat']
            # mask = DIFF_df.iloc[indD]['Mask_HDU'].data
            # noise = DIFF_df.iloc[indD]['Noise_HDU'].data
            try:
                mask = DIFF_df.iloc[indD]['Mask_HDU'].data
                noise = DIFF_df.iloc[indD]['Noise_HDU'].data
            except Exception as e:
                #print('\n***ERROR:*** '+str(e))
                mask = np.zeros(HDU.data.shape)
                noise = np.zeros(HDU.data.shape)
            
            NoiseMAT = noise*1.0/LEplots.im_norm_factor(HDU)
            NoiseMAT[~mask2bool(mask)] = np.nan
            avg_noise = np.nanmean(NoiseMAT)
            
            zptmag = DIFF_df.iloc[indD]['ZPTMAG']
            if zptmag is not None:
                FP_df.iloc[indD]['ZPTMAG'] = zptmag
            else:
                FP_df.iloc[indD]['ZPTMAG'] = 0
            w = wcs.WCS(header)
            
            if indS==0:
                image = DIFF_df.iloc[indD]['Diff_HDU'].data*1.0
                # Zx = ndimage.sobel(image,axis=1)/8
                # Zy = ndimage.sobel(image,axis=0)/8
                # grad = np.sqrt(Zx**2+Zy**2)
                # grad_thres=np.percentile(grad,99.5)
                # boolinds_mat = (ndimage.median_filter((grad>grad_thres)*1.0,size=3)==1)
                # image[boolinds_mat] = np.nan
                images.append(image.copy())
                
#                inds_mask = np.ravel_multi_index(np.where(mask2bool(mask)),mask.shape)
#                inds_mask_lst.append(inds_mask.copy())
            else:
                image = images[indD].copy()
#                inds_mask = inds_mask_lst[indD].copy()
                
            
            if PSFeq_overFWHM is not None:
                if indS==0:
                    image = matchFWHM(DIFF_df,indD,maxFWHM+PSFeq_overFWHM)
                    images.append(image.copy())
                else:
                    image = images[indD].copy()
#                fwhm = DIFF_df.iloc[indD]['FWHM'] # [pixels]
#                print(fwhm)
#                gss_FWHM = np.sqrt(((ang2pix(w,maxFWHM+PSFeq_overFWHM))**2)-(fwhm**2))
#                print(gss_FWHM)
#                image = gaussian_filter(image, sigma=gss_FWHM/(2*np.sqrt(2*np.log(2))))
    
            FPcntr_pix, FP_pa_pix_uv, FPlen_pix = FPparams_world2pix(w, FPcntr_world, FP_pa_world, FPlen_world)
            if width is None:
                width_pix = 1
                FP_df.iloc[indD]['WrldWidth'] = pix2ang(w, width_pix)
            else:
                width_pix = ang2pix(w, width)
                FP_df.iloc[indD]['WrldWidth'] = width
            if uniform_wcs and indD!=0:
                flux, err, pixVec, pixProj, pixPerp, pixCorners, boolind_mat_wcs, Xmat, Ymat, dotprod_mat, pixCntrs, boolind_mat = MatLinSamp(image, mask, noise, FPcntr_pix, FP_pa_pix_uv, FPlen_pix, width_pix, N=int(FPlen_world.arcsec), samp_mask_pre_calc = boolind_mat)
            else:
                flux, err, pixVec, pixProj, pixPerp, pixCorners, boolind_mat_wcs, Xmat, Ymat, dotprod_mat, pixCntrs, boolind_mat = MatLinSamp(image, mask, noise, FPcntr_pix, FP_pa_pix_uv, FPlen_pix, width_pix, N=int(FPlen_world.arcsec))
            
            # flux, err, pixVec, pixProj, pixPerp, pixCorners, boolind_mat_wcs, Xmat, Ymat, dotprod_mat, pixCntrs = MatLinSamp(image, mask, noise, FPcntr_pix, FP_pa_pix_uv, FPlen_pix, width_pix, N=N_bins)
#            if indD==0:
#                flux, err, pixVec, pixProj, pixCorners, boolind_mat_wcs, Xmat, Ymat, dotprod_mat = MatLinSamp(image, mask, noise, FPcntr_pix, FP_pa_pix_uv, FPlen_pix, width_pix, N=N_bins)
#                inds_wcs = np.ravel_multi_index(np.where(boolind_mat_wcs),boolind_mat_wcs.shape)
#                inds_intersect = inds_wcs[mask2bool(mask).flatten()[inds_wcs]]
##                if indS==0:
##                    inds_intersect = np.intersect1d(inds_wcs,inds_mask,assume_unique=True)
##                boolind_mat = np.logical_and(mask2bool(mask),boolind_mat_wcs)
#            else:
##                inds_intersect = np.intersect1d(inds_wcs,inds_mask,assume_unique=True)
##                boolind_mat = np.logical_and(mask2bool(mask),boolind_mat_wcs)
#                if np.any(inds_wcs):
#                    inds_intersect = inds_wcs[mask2bool(mask).flatten()[inds_wcs]]
#                    flux = image.flatten()[inds_intersect]#[boolind_mat]
#                    err = noise.flatten()[inds_intersect]#[boolind_mat]
#                    pixVec = np.concatenate((np.reshape(Xmat.flatten()[inds_intersect],(flux.size,1)),np.reshape(Ymat.flatten()[inds_intersect],(flux.size,1))),axis=1)
#                    pixProj = dotprod_mat.flatten()[inds_intersect]#[boolind_mat]
#                else:
#                    flux, err, pixVec, pixProj = [None,None,None,None]
            
            FP_df.iloc[indD]['PixVec'] = pixVec
            FP_df.iloc[indD]['FluxProfile_ADU'] = flux
            FP_df.iloc[indD]['NoiseProfile_ADU'] = err
            wrldCorners = w.wcs_pix2world(pixCorners,1)
            wrldCntrs = w.wcs_pix2world(np.array(pixCntrs).reshape((len(pixCntrs),2)),1)
            
            if flux is not None:
                wrldVec = w.wcs_pix2world(pixVec,1)
                wrldProj = pix2ang(w,pixProj)
                wrldPerp = pix2ang(w,pixPerp)
                if REF_image is not None:
#                    print(header['KSUM00'])
                    flux_cal = (flux/LEplots.im_norm_factor(HDU)) - REF_image.flatten()[inds_intersect]#[boolind_mat]
#                    print(np.sum(~np.isnan(flux_cal)))
#                    flux_cal = flux*np.power(10,-zptmag/2.5) - REF_image[boolind_mat]
                else:
                    flux_cal = flux/LEplots.im_norm_factor(HDU)#*np.power(10,-zptmag/2.5)
                err_cal = err/LEplots.im_norm_factor(HDU)
                flux_mag = -2.5*np.log10(flux) + FP_df.iloc[indD]['ZPTMAG']
                xx, yy, yyerr, yBref, stdBref, binCnt = prof_bins(wrldProj.arcsec,flux_cal,FPlen_world.arcsec,N=int(FPlen_world.arcsec),OL=bin_OL)
                err_xx, err_yy, err_yyerr, err_yBref, err_stdBref, err_binCnt = prof_bins(wrldProj.arcsec,err_cal,FPlen_world.arcsec,N=int(FPlen_world.arcsec),OL=bin_OL)
                # xx, yy, yyerr, yBref, stdBref, binCnt = prof_bins(wrldProj.arcsec,flux_cal,FPlen_world.arcsec,N=N_bins,OL=bin_OL)
                # err_xx, err_yy, err_yyerr, err_yBref, err_stdBref, err_binCnt = prof_bins(wrldProj.arcsec,err_cal,FPlen_world.arcsec,N=N_bins,OL=bin_OL)
                
                
#                abnormal_inds = yyerr>(3*err_yy/np.sqrt(binCnt))
#                abnormal_inds = np.logical_or(abnormal_inds,binCnt<0.5*np.nanmax(binCnt))
#                abnormal_inds = binCnt<0.5*np.nanmax(binCnt)
##                print(np.sum(abnormal_inds)/yyerr.size)
#                yyerr[abnormal_inds] = np.nan
#                yy[abnormal_inds] = np.nan
#                abnormal_inds = yyerr>3*(get_sig_adu_cal(DIFF_df,indD)/np.sqrt(binCnt))
#                xx[abnormal_inds] = np.nan
#                yy[abnormal_inds] = np.nan
#                yyerr[abnormal_inds] = np.nan
#                print('xx: '+str(np.sum(~np.isnan(xx))))
#                print('yy: '+str(np.sum(~np.isnan(yy))))
#                print('yyerr: '+str(np.sum(~np.isnan(yyerr))))
            else:
                wrldVec, wrldProj, flux_cal, flux_mag = [None, None, None, None]
                xx, yy, yyerr, yBref, stdBref, binCnt = [None, None, None, None, None, None]
            
            FP_df.iloc[indD]['WrldVec'] = wrldVec
            FP_df.iloc[indD]['ProjAng'] = wrldProj
            FP_df.iloc[indD]['PerpAng'] = wrldPerp
            FP_df.iloc[indD]['FluxProfile_ADUcal'] = flux_cal
            FP_df.iloc[indD]['FluxProfile_MAG'] = flux_mag
            FP_df.iloc[indD]['WrldCorners'] = wrldCorners
            FP_df.iloc[indD]['WrldCntrs'] = wrldCntrs
            
            FP_df.iloc[indD]['FP_Ac_bin_x'] = xx
            FP_df.iloc[indD]['FP_Ac_bin_y'] = yy#err_yy/np.sqrt(binCnt)
            FP_df.iloc[indD]['FP_Ac_bin_yerr'] = yyerr
            FP_df.iloc[indD]['FP_Ac_bin_yBref'] = yBref
            FP_df.iloc[indD]['FP_Ac_bin_ystdBref'] = stdBref
            FP_df.iloc[indD]['FP_Ac_bin_Cnt'] = binCnt
            
        FP_df_lst.append(FP_df.copy())
        
        
    return FP_df_lst



def MatLinSamp(image, mask, noise, FPcntr_pix, FP_pa_pix_uv, FPlen_pix, width_pix, N=50, samp_mask_pre_calc = None):
    
    # for convenience, here we internally reshape FPcntr_pix and FP_pa_pix_uv from (1,2)=[[x,y]] to (2,)=[x,y]
    FPcntr_pix = np.reshape(FPcntr_pix,(2,))
    FP_pa_pix_uv = np.reshape(FP_pa_pix_uv,(2,))
    FP_pa_pix_uv_perp = np.array([-FP_pa_pix_uv[1],FP_pa_pix_uv[0]])
    
    pixCorners = np.array([FPcntr_pix -FP_pa_pix_uv*FPlen_pix/2 +FP_pa_pix_uv_perp*width_pix/2,
                           FPcntr_pix +FP_pa_pix_uv*FPlen_pix/2 +FP_pa_pix_uv_perp*width_pix/2,
                           FPcntr_pix +FP_pa_pix_uv*FPlen_pix/2 -FP_pa_pix_uv_perp*width_pix/2,
                           FPcntr_pix -FP_pa_pix_uv*FPlen_pix/2 -FP_pa_pix_uv_perp*width_pix/2,
                           FPcntr_pix -FP_pa_pix_uv*FPlen_pix/2 +FP_pa_pix_uv_perp*width_pix/2])
    pixCntrs = []
    for i in np.arange(N):
        pixCorners = np.concatenate((pixCorners,np.array([pixCorners[4] +i*FP_pa_pix_uv*FPlen_pix/N],ndmin=2)))
        pixCorners = np.concatenate((pixCorners,np.array([pixCorners[-1] -FP_pa_pix_uv_perp*width_pix],   ndmin=2)))
        pixCorners = np.concatenate((pixCorners,pixCorners[pixCorners.shape[0]-2:pixCorners.shape[0]-1]))
        pixCntrs.append(np.array([pixCorners[4] +(i+0.5)*FP_pa_pix_uv*FPlen_pix/N -0.5*FP_pa_pix_uv_perp*width_pix],ndmin=2))
    
    Xmat, Ymat = np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[0]))
    
    v_X = Xmat - FPcntr_pix[0]
    v_Y = Ymat - FPcntr_pix[1]
    
    dotprod_mat = v_X*FP_pa_pix_uv[0] + v_Y*FP_pa_pix_uv[1]
    
    dist_mat = v_X*FP_pa_pix_uv_perp[0] + v_Y*FP_pa_pix_uv_perp[1]
    # distvec_X = v_X - dotprod_mat*FP_pa_pix_uv[0]
    # distvec_Y = v_Y - dotprod_mat*FP_pa_pix_uv[1]
    # dist_mat = np.sqrt(distvec_X**2 + distvec_Y**2)
    
    if samp_mask_pre_calc is not None:
        boolind_mat = samp_mask_pre_calc
        boolind_mat_wcs = samp_mask_pre_calc
    else:
        boolind_mat_wcs = np.logical_and(np.abs(dist_mat)<(width_pix/2), np.abs(dotprod_mat)<(FPlen_pix/2))
        boolind_mat = np.logical_and(mask2bool(mask),boolind_mat_wcs)
    
    if np.any(boolind_mat.flatten()):
        flux = image[boolind_mat]
        err = noise[boolind_mat]
        pixVec = np.concatenate((np.reshape(Xmat[boolind_mat],(flux.size,1)),np.reshape(Ymat[boolind_mat],(flux.size,1))),axis=1)
        pixProj = dotprod_mat[boolind_mat]
        pixPerp = dist_mat[boolind_mat]
    else:
        flux, err, pixVec, pixProj = [None, None, None, None]
    
    return flux, err, pixVec, pixProj, pixPerp, pixCorners, boolind_mat_wcs, Xmat, Ymat, dotprod_mat, pixCntrs, boolind_mat



def FPparams_world2pix(w, FPcntr_world, FP_pa_world, FPlen_world):
    
    FPcntr_pix = w.wcs_world2pix(np.array([[FPcntr_world.ra.deg, FPcntr_world.dec.deg]]), 1)
    
    FPlen_pix = ang2pix(w, FPlen_world)
    
    RA_uv = FPcntr_world.spherical.unit_vectors()['lon']
    DEC_uv = FPcntr_world.spherical.unit_vectors()['lat']
    
    FPcntr_world_car = FPcntr_world.represent_as(CartesianRepresentation)
    FPcntr_pa_US = (FPcntr_world_car + pix2ang(w,1).rad*(np.sin(FP_pa_world.rad)*RA_uv + np.cos(FP_pa_world.rad)*DEC_uv)).represent_as(UnitSphericalRepresentation)
    FP_pa_pix_uv = normalize(w.wcs_world2pix(np.array([[FPcntr_pa_US.lon.deg,FPcntr_pa_US.lat.deg]]),1) - FPcntr_pix)
    
    return FPcntr_pix, FP_pa_pix_uv, FPlen_pix



def mask2bool(mask,mode='safe'): 
    if mode=='safe':
        boolmat = mask==0 # only "safe" pixels
    elif mode=='suspicious':
        boolmat = np.logical_or(mask==0,mask<0x8000) # permit "suspicious" pixels also
    elif mode=='suspicious-only':
        boolmat = np.logical_and(mask!=0,mask<0x8000) # permit only "suspicious" pixels, without "safe" pixels
    # boolmat=True represents GOOD pixels (boolmat is inteded to be used for image matrix indexing)
    return boolmat

def addPeakLoc(FP_df_lst,method='argmax'):
    
    for k in np.arange(len(FP_df_lst)):
        print('####====####')
        print('k='+str(k))
        FP_df_lst[k].insert(FP_df_lst[k].shape[1],'Peak_ProjAng_arcsec',None)
        if method=='argmax':
            col_ind = FP_df_lst[k].columns.get_loc('Peak_ProjAng_arcsec')
            for i in np.arange(len(FP_df_lst[k])):
                if FP_df_lst[k].iloc[i]['FP_Ac_bin_y'] is None:
                    continue
                print('i='+str(i))
                print(FP_df_lst[k].iloc[i]['FP_Ac_bin_y'].shape)
                print(np.nanargmax(FP_df_lst[k].iloc[i]['FP_Ac_bin_y']))
                print(FP_df_lst[k].iloc[i]['FP_Ac_bin_x'][np.nanargmax(FP_df_lst[k].iloc[i]['FP_Ac_bin_y'])])
                print('====')
                FP_df_lst[k].iat[i,col_ind] = FP_df_lst[k].iloc[i]['FP_Ac_bin_x'][np.nanargmax(FP_df_lst[k].iloc[i]['FP_Ac_bin_y'])]
        
        elif method=='something_else':
            pass
    
    return FP_df_lst

def prof_bins(x,y,FP_len,N=100,OL=0.0):
    if not (OL>=0 and OL<100):
        OL=0
#    OL = np.nanmax(np.array([0.0,np.nanmin(np.array([0.95,OL]))]))
#    stats.binned_statistic(phases_shft,LinFlux_p,statistic='mean',bins=1+int((max_phs-min_phs)/phs_res),range=(phs_bin_edges1[0]+np.mean(np.diff(phs_bin_edges1))/2,phs_bin_edges1[-1]))
    xB = []#np.zeros((N,))
    yB = []#np.zeros((N,))
    stdB = []#np.zeros((N,))
    binCnt = []
    
    yBref = np.zeros(y.shape)
    stdBref = np.zeros(y.shape)
    
    x_left = -FP_len/2#np.nanmin(x)
    x_right = FP_len/2#np.nanmax(x)
    bin_size = 1.0*(x_right-x_left)/N
    i=0
    
    bin_right=x_left
    while bin_right<x_right:
        bin_left = x_left +i*bin_size*(1.0-OL)
        bin_right = bin_left +bin_size
        xB.append(np.mean((bin_left,bin_right)))
        inds = np.logical_and(x>=bin_left,x<bin_right)
        vals = y[inds]
        binCnt.append(vals.size)
        yB.append(np.nanmedian(vals))
        stdB.append(np.nanstd(vals)/np.sqrt(vals.size))
        yBref[inds] = yB[i]
        stdBref[inds] = stdB[i]
        i=i+1
    xB = np.array(xB)
    yB = np.array(yB)
    stdB = np.array(stdB)
    binCnt = np.array(binCnt)
    
    return xB, yB, stdB, yBref, stdBref, binCnt

def mask_intersect(DIFF_df, inds=None):
    if inds is None:
        inds = np.arange(len(DIFF_df))
    
    maskColInd = DIFF_df.columns.get_loc('Mask_mat')
    tmp_mask = DIFF_df.iloc[inds[0],maskColInd]
    for ind in inds[1:]:
        tmp_mask[np.logical_not(np.logical_and(mask2bool(tmp_mask),mask2bool(DIFF_df.iloc[ind,maskColInd])))] = 0x8000
    
    for ind in inds:
        DIFF_df.iat[ind,maskColInd] = tmp_mask
    return DIFF_df

def LE_DF_xyz(DIFF_df,SN_sc,SN_time,D_ly):
    DIFF_df.insert(DIFF_df.shape[1],'xyz_list',[ [] for i in range(len(DIFF_df)) ])
    for i in np.arange(len(DIFF_df)):
        t_LE=Time(DIFF_df.iloc[i]['Idate'],format='mjd')
        for coord in DIFF_df.iloc[i]['coords_list']:
            coord = SkyCoord(coord[0][0], coord[0][1], frame='fk5', unit='deg')
            sep = coord.separation(SN_sc)
            PA = coord.position_angle(SN_sc)
            
            delta_t_years = (t_LE - SN_time).to(u.year).value
            rho, z = rho_t_D_angSep(delta_t_years,D_ly,sep.deg)
            x = rho*np.cos(PA.rad)
            y = rho*np.sin(PA.rad)
            DIFF_df.iloc[i]['xyz_list'].append(np.array([x,y,z],ndmin=2))
    return DIFF_df


def FP_Xcorr(FP_df_lst,DIFF_df):
    FP_Xcorr_df_lst=[]
    FP_Xcorr_argmat_lst=[]
    for FP_df in FP_df_lst:
        FP_Xcorr_df = pd.DataFrame(index=FP_df.index, columns=FP_df.index)
        FP_Xcorr_argmat = np.empty((len(FP_df),len(FP_df),4))
        FP_Xcorr_argmat[:] = np.nan
        ref_FP = 0#np.nanmedian(np.stack(FP_df['FP_Ac_bin_y'].to_numpy()),axis=0)
        for i in np.arange(len(FP_df)):
            y1 = FP_df.iloc[i]['FP_Ac_bin_y'].copy()
            if y1 is None:
                continue
#            sig_adu_cal =  get_sig_adu_cal(DIFF_df,i)
            y1=y1-ref_FP
            y1[np.isnan(y1)]=0
#            print('i='+str(i))
#            y1[y1<2*sig_adu_cal]=0
            for k in np.arange(len(FP_df)):
                y2 = FP_df.iloc[k]['FP_Ac_bin_y'].copy()
                if y2 is None:
                    continue
#                sig_adu_cal =  get_sig_adu_cal(DIFF_df,k)
                y2=y2-ref_FP
                y2[np.isnan(y2)]=0
#                print('k='+str(k))
#                y2[y2<2*sig_adu_cal]=0
                y1 = (y1 - np.mean(y1)) / (np.std(y1) * len(y1))
                y2 = (y2 - np.mean(y2)) / (np.std(y2))
                xcorr = np.correlate(y1,y2,mode='same')
                FP_Xcorr_df.iat[i,k] = xcorr
                FP_Xcorr_argmat[i,k,0] = np.nanargmax(xcorr)
                FP_Xcorr_argmat[i,k,1] = xcorr[FP_Xcorr_argmat[i,k,0].astype(int)]
                x = FP_df.iloc[i]['FP_Ac_bin_x']
                FP_Xcorr_argmat[i,k,2] = x[FP_Xcorr_argmat[i,k,0].astype(int)]
                FP_Xcorr_argmat[i,k,3] = DIFF_df.iloc[k]['Idate'] - DIFF_df.iloc[i]['Idate']
        FP_Xcorr_df_lst.append(FP_Xcorr_df.copy())
        FP_Xcorr_argmat_lst.append(FP_Xcorr_argmat)
    return FP_Xcorr_df_lst, FP_Xcorr_argmat_lst

def get_sig_adu_cal(DIFF_df,ind):
    HDU = DIFF_df.iloc[ind]['Diff_HDU']
    zpt_lin = get_zpt_lin(DIFF_df,ind)
    sig_adu_cal = HDU.header['SKYSIG']*zpt_lin
    return sig_adu_cal

def get_zpt_lin(DIFF_df,ind):
    zptmag = DIFF_df.iloc[ind]['ZPTMAG']
    zpt_lin = np.power(10,-zptmag/2.5)
    return zpt_lin

# ==== UTILITIES ====
    
def ang2pix(w, ang):
    ang_pix = Angle(wcs.utils.proj_plane_pixel_scales(w)[0],w.wcs.cunit[0])
    pix = ang.deg/ang_pix.deg
    return pix


def pix2ang(w, pix):
    ang_pix = Angle(wcs.utils.proj_plane_pixel_scales(w)[0],w.wcs.cunit[0])
    ang = ang_pix*pix
    return ang


def scalarize(obj):
    if obj.isscalar is False:
        scalar_obj = obj[0]
    else:
        scalar_obj = obj
    return scalar_obj


def normalize(vec):
    norma = v_norma(vec)
    n_vec = vec/norma
    return n_vec


def v_norma(vec):
    SQ = np.squeeze
    norma = np.sqrt(np.dot(SQ(vec),SQ(vec)))
    return norma

# ==== MATH ====
#def AngSep(LE,SN):
#    
#    return AngSep
    
def z_t_D_rho(delta_t_years,D_ly,rho_ly):
    z = 0.5*D_ly - 0.5*(delta_t_years+D_ly)*np.sqrt(1-((4/(2*delta_t_years*D_ly + delta_t_years**2))*(rho_ly**2)))
#    z = 0.5*((rho_ly**2)/delta_t_years) - 0.5*delta_t_years
    return z

def rho_t_D_angSep(delta_t_years,D_ly,angSep_deg):
    angSep_rad = angSep_deg*np.pi/180
    angSep_tan = np.tan(angSep_rad)
    sum_t_D_sq = (delta_t_years+D_ly)**2
    
    a = (4/(angSep_tan**2)) + 4*(sum_t_D_sq/(sum_t_D_sq - D_ly**2))
    b = -4*D_ly/angSep_tan
    c = D_ly**2 - sum_t_D_sq
    
    disc = np.sqrt(b**2 -4*a*c)
    rho1 = (-b + disc)/(2*a)
    rho2 = (-b - disc)/(2*a)
#    print('rho1:'+str(rho1))
#    print('rho2:'+str(rho2))
    
    rho = np.max([rho1,rho2])
    z = D_ly - rho/angSep_tan
    return rho, z