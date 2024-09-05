#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:34:21 2019

@author: roeepartoush
"""


import numpy as np
#from astropy.modeling import models, fitting
#import pandas as pd
#import astropy.units as u

from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

#import LeTools_Module as LeT

from inspect import signature
import matplotlib.pyplot as plt

def FitFluxProfile(DIFF_df,FP_df,FP_df_PSFeq,LC_fit_params,slope,intercept):
    
    LCtable_f      = LC_fit_params['LCtable']
    dm15_f         = LC_fit_params['dm15']
    phs_shft_DOF_f = LC_fit_params['phs_shft_DOF']
    ref_FWHM_phs_f = LC_fit_params['ref_FWHM_phs']
    
    for i in np.arange(len(FP_df)):
        print('### '+str(i)+' ###')
        FWHM_phs = DIFF_df.iloc[i]['FWHM_ang'].arcsec*slope
        LC_fit_func = GenConvLC(LCtable=LCtable_f, dm15=dm15_f, phs_shft_DOF=phs_shft_DOF_f, ref_FWHM_phs=np.sqrt(ref_FWHM_phs_f**2 + FWHM_phs**2))
        
        params_lst = list(signature(LC_fit_func).parameters)
        params_lst.remove('x')
        if 'dm15_param' in params_lst:
            dm15_DOF = True
            par_init = [50, 500, 0.7]
            bnd = (np.array([0, 0, 0]), np.array([100, 1e3, np.inf]))
        else:
            dm15_DOF = False
            if 'phs_shft' in params_lst:
                phs_shft_DOF = True
                par_init = [50, 500,0]
                bnd = (np.array([0, 0, -np.inf]), np.array([100, 1e3, np.inf]))
            else:
                phs_shft_DOF = False
                par_init = [10, 100]
                bnd = (np.array([0, 0]), np.array([100, 1e3]))
        
        bnd[0][0] = 0*FWHM_phs
#        print(str(i)+' '+str(FWHM_phs))
        x = -(np.array(FP_df.iloc[i]['ProjAng'].arcsec)-FP_df_PSFeq.iloc[i]['Peak_ProjAng_arcsec'])*slope
        flux_scale = Mag2ADU(DIFF_df.iloc[i]['ZPTMAG'])*1e12
        y = flux_scale*np.array(FP_df.iloc[i]['FluxProfile'])
        z = flux_scale*np.array(FP_df.iloc[i]['NoiseProfile'])
        
        x = x[y>-z]
        ytmp = y[y>-z]
        z = z[y>-z]
        y = ytmp
        
        popt, pcov = curve_fit(LC_fit_func, x, y, bounds=bnd, p0 = par_init)
        plt.scatter(x,y,s=1,label=str(i))
        
        x_eq = -(np.array(FP_df_PSFeq.iloc[i]['ProjAng'].arcsec)-FP_df_PSFeq.iloc[i]['Peak_ProjAng_arcsec'])*slope
        y_eq = flux_scale*np.array(FP_df_PSFeq.iloc[i]['FluxProfile'])
        plt.scatter(x_eq,y_eq,s=1,label='kaka')
        plt.gca().legend(prop={'size': 6})
        
        FP_df.loc[i,'fitTotWid_phs'] = popt[params_lst.index('phs_wid')]
        FP_df.loc[i,'fitPeakFlux_phs'] = popt[params_lst.index('PeakFlux')]
        if dm15_DOF:
            FP_df.loc[i,'fitDM15'] = popt[params_lst.index('dm15_param')]
        elif phs_shft_DOF:
            FP_df.loc[i,'fit_phs_shft'] = popt[params_lst.index('phs_shft')]
        
        FP_df.loc[i,'fitDstWid_phs'] = np.sqrt(popt[params_lst.index('phs_wid')]**2 - FWHM_phs**2)
        FP_df.loc[i,'fit_LC_fit_func'] = LC_fit_func
    return FP_df


def GenConvLC(LCtable, dm15=None, phs_shft_DOF=False, ref_FWHM_phs=None):
    plt.figure()
    phs_res = 0.1 # 0.1 days phase resolution should be enough for all practical uses
    dm15_lst = np.asarray(LCtable['dm15'])
    if dm15 is None:
        print('###DM15!!!###')
        def ConvLC(x, phs_wid, PeakFlux, dm15_param):
            tbl_ind = (np.abs(dm15_lst - dm15_param)).argmin()#np.nonzero(LCtable['dm15']==dm15_param)[0][0]
            LCfunc = LCtable['func_L'][tbl_ind]
            phases_p = LCtable['phases'][tbl_ind]
            
            y_p_unscaled, x_p = LCgrid_eval(LCfunc,phs_wid,phs_res,phases_p)
            
            ref_FWHM_phshft = 0
            if ref_FWHM_phs:
                y_p_max_FWHM, x_p_max_FWHM = LCgrid_eval(LCfunc,np.sqrt(ref_FWHM_phs**2+phs_wid**2),phs_res,phases_p)
                ref_FWHM_phshft = x_p_max_FWHM[np.nanargmax(y_p_max_FWHM)]
            
            const = PeakFlux/LCfunc(0) # normalize maximun luminosity to PeakFlux
            x_arr=np.array(x)
            x_arr = x_arr+0*x_p[np.nanargmax(y_p_unscaled)] +ref_FWHM_phshft
            y = np.interp(x_arr, x_p, const*y_p_unscaled, left=0, right=0)
            return y
        
    else:
        tbl_ind = (np.abs(dm15_lst - dm15)).argmin()#np.nonzero(LCtable['dm15']==dm15)[0][0]
        LCfunc = LCtable['func_L'][tbl_ind]
        phases_p = LCtable['phases'][tbl_ind]
        if phs_shft_DOF:
            print('###phs_shft_DOF!!!###')
            def ConvLC(x, phs_wid, PeakFlux, phs_shft):
                
                y_p_unscaled, x_p = LCgrid_eval(LCfunc,phs_wid,phs_res,phases_p)
                
                ref_FWHM_phshft = 0
                if ref_FWHM_phs:
                    y_p_max_FWHM, x_p_max_FWHM = LCgrid_eval(LCfunc,np.sqrt(ref_FWHM_phs**2+phs_wid**2),phs_res,phases_p)
                    ref_FWHM_phshft = x_p_max_FWHM[np.nanargmax(y_p_max_FWHM)]
                
                const = PeakFlux/LCfunc(0) # normalize maximun luminosity to PeakFlux
                x_arr=np.array(x)
                x_arr = x_arr + phs_shft
                x_arr = x_arr+0*x_p[np.nanargmax(y_p_unscaled)] +ref_FWHM_phshft
                y = np.interp(x_arr, x_p, const*y_p_unscaled, left=0, right=0)
                plt.plot(np.array(x),y,label=('phs_wid='+str(int(phs_wid))+', PeakFlux='+str(int(PeakFlux))+', phs_shft='+str(int(phs_shft))+', ref_FWHM_phshft='+str(int(ref_FWHM_phshft))))
                return y
            
        else:
            print('###SIMPLE!!!###')
            def ConvLC(x, phs_wid, PeakFlux):
                y_p_unscaled, x_p = LCgrid_eval(LCfunc,phs_wid,phs_res,phases_p)
                
                ref_FWHM_phshft = 0
                if ref_FWHM_phs:
#                    print('ref_FWHM_phs = '+str(ref_FWHM_phs)+'\n')
#                    print('phs_wid = '+str(phs_wid)+'\n')
                    y_p_max_FWHM, x_p_max_FWHM = LCgrid_eval(LCfunc,np.sqrt(ref_FWHM_phs**2+phs_wid**2),phs_res,phases_p)
                    ref_FWHM_phshft = x_p_max_FWHM[np.nanargmax(y_p_max_FWHM)]
                
                const = PeakFlux/LCfunc(0) # normalize maximun luminosity to PeakFlux
                x_arr=np.array(x)
                x_arr = x_arr+0*x_p[np.nanargmax(y_p_unscaled)] +ref_FWHM_phshft
                y = np.interp(x_arr, x_p, const*y_p_unscaled, left=0, right=0)
#                plt.plot(np.array(x),y,label=('phs_wid='+str(int(phs_wid))+', PeakFlux='+str(int(PeakFlux))+', ref_FWHM_phshft='+str(int(ref_FWHM_phshft))))
                return y
        
#        def ConvLC(x, phs_wid, PeakFlux):
#            x=list(x)
#            y=x.copy()
#            sigma = phs_wid/FWHMoSIG
#            const = PeakFlux/Mag2ADU(LCfunc(0)) # normalize maximun luminosity to PeakFlux
#            N = np.nanmax([1+np.round(6*sigma/phs_res),100])
#    
#            rel_e_grid = np.linspace(-3*sigma,3*sigma,N)
#            res = np.nanmean(np.diff(rel_e_grid))
#            for i in np.arange(len(x)):
#                eval_grid = x[i] + rel_e_grid
#                y[i] = const*gaussian_filter(Mag2ADU(LCfunc(eval_grid)),sigma/res)[np.round(N/2).astype(int)]
#            return np.array(y)
    return ConvLC


def LCgrid_eval(LCfunc,phs_wid,phs_res,phases_p):
    FWHMoSIG = 2*np.sqrt(2*np.log(2))
    
    phs_sigma = phs_wid/FWHMoSIG
    max_ph = np.nanmax(phases_p)+phs_sigma*3
    min_ph = np.nanmin(phases_p)-phs_sigma*3
#    print('max_ph = '+str(max_ph)+'\n')
#    print('min_ph = '+str(min_ph)+'\n')
    x_p = np.linspace(min_ph,max_ph,np.ceil((max_ph-min_ph)/phs_res).astype(int)+1)
    res = np.nanmean(np.diff(x_p))
    y_p_unscaled = gaussian_filter(LCfunc(x_p), sigma=phs_sigma/res, mode='constant', cval=0)
    
    return y_p_unscaled, x_p


def Mag2ADU(x, inverse=False):
    if inverse:
        y = -2.5*np.log10(x)
    else:
        y = np.power(10,-x/2.5)
    return y


def info(x):
    tp = type(x)
    shp = x.shape
    print('Type: ' +str(tp)+'\n')
    print('Shape: '+str(shp)+'\n')
    return