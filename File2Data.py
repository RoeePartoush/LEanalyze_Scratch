#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:16:07 2019

@author: roeepartoush
"""
import re
from io import StringIO
import numpy as np
import pandas as pd
import astropy.io.ascii as asc
from astropy.table import Table, Column
from astropy.io import fits
import DustFit as DF
import os
from astropy import wcs
import LEtoolbox as LEtb
from scipy.ndimage import gaussian_filter
from scipy import stats
from astropy import wcs
SQ = np.squeeze

def FitsDiff(diff_flnm):
    # diff_flnm = list of strings containing full absolute path EXCEPT extension (without the '.fits')
    
    DFclmn = ['filename','Idate','Tdate','Diff_HDU','Mask_HDU','Mask_mat','Noise_HDU','Noise_mat','FWHM','ZPTMAG','FWHM_ang','M5SIGMA','WCS_w']
    difDF = pd.DataFrame(index=np.arange(len(diff_flnm)),columns=DFclmn)
    
    for ind in np.arange(len(diff_flnm)):
        print('\n***FitsDiff*** iter. no '+str(ind))
        difDF.at[ind,'filename'] = os.path.basename(diff_flnm[ind])
        print(diff_flnm[ind] + '.fits')
        Diff_hdu = fits.open(diff_flnm[ind])[0]
        difDF.loc[ind]['Diff_HDU'] = Diff_hdu
        difDF.loc[ind]['WCS_w'] = wcs.WCS(Diff_hdu.header)
        try:
            difDF.loc[ind]['Mask_HDU'] = fits.open(diff_flnm[ind] + '.mask.fits')[0]
            # difDF.at[ind,'Mask_mat'] = fits.getdata(diff_flnm[ind] + '.mask.fits')
            difDF.loc[ind]['Noise_HDU'] = fits.open(diff_flnm[ind] + '.noise.fits')[0]
            # difDF.at[ind,'Noise_mat'] = fits.getdata(diff_flnm[ind] + '.noise.fits')
        except Exception as e:
            print('\n***ERROR:*** '+str(e))
            # difDF.at[ind,'Mask_mat'] = np.zeros(Diff_hdu.data.shape)
            # difDF.at[ind,'Noise_mat'] = np.zeros(Diff_hdu.data.shape)
            pass
        print(ind)
        # difDF.loc[ind]['Idate'] = Diff_hdu.header['MJD-OBS']
        
    difDF = FitsParams(diff_flnm, difDF)
    difDF.insert(difDF.shape[1],'coords_list',[ [] for i in range(len(difDF)) ])
    difDF.sort_values('Idate',inplace=True)
    difDF.reset_index(drop=True,inplace=True)
    print('\n====DONE.====')
    return difDF

def FitsParams(diff_flnm, difDF):
    for ind in np.arange(len(diff_flnm)):
        print('\n***FitsParams*** iter. no '+str(ind))
        try:
            try:
                Iflnm = os.path.dirname(diff_flnm[ind]) + '/' + difDF.iloc[ind]['Diff_HDU'].header['IMNAME']
                Tflnm = os.path.dirname(diff_flnm[ind]) + '/' + difDF.iloc[ind]['Diff_HDU'].header['TMPLNAME']
                print('Found Image & Template .dcmp!\n')
            except Exception as e:
                print('\n***ERROR:*** '+str(e))
                try:
                    Iflnm = os.path.dirname(diff_flnm[ind]) + '/' + os.path.basename(difDF.iloc[ind]['Diff_HDU'].header['TARGET'])
                    Tflnm = os.path.dirname(diff_flnm[ind]) + '/' + os.path.basename(difDF.iloc[ind]['Diff_HDU'].header['TEMPLATE'])
                    print('Found Image & Template .dcmp!\n')
                except Exception as e:
                    print('\n***ERROR:*** '+str(e))
                    print('\nImage & Template .dcmp not found. Sorry!')
            Iflnm = Iflnm[0:len(Iflnm)-5] # remove the '.fits' extension
            Tflnm = Tflnm[0:len(Tflnm)-5] # remove the '.fits' extension
            IdcmpH = fits.open(Iflnm + '.dcmp')[0].header
            TdcmpH = fits.open(Tflnm + '.dcmp')[0].header
            difDF.iloc[ind]['M5SIGMA'] = IdcmpH['M5SIGMA']
            print('something!!!!!$$$$$$$$')
            print(IdcmpH['M5SIGMA'])
            print(difDF.iloc[ind]['M5SIGMA'])
        except Exception as e:
            pass
        try:
            dcmpH = fits.open(diff_flnm[ind] + '.dcmp')[0].header
            difDF.iloc[ind]['FWHM'] = dcmpH['FWHM']
            difDF.iloc[ind]['M5SIGMA'] = dcmpH['M5SIGMA']
#            difDF.iloc[ind]['M5SIGMA'] = fishKW(dcmpH,['M5SIGMA'])
            difDF.iloc[ind]['ZPTMAG'] = fishKW(dcmpH,['ZPTMAG00','ZPTMAG','MAGZERO'])
        except Exception as e:
            print('\nWARNING! .dcmp file for '+diff_flnm[ind]+' not found.\nTrying to find Image & Template .dcmp')
            print('\n***ERROR:*** '+str(e))
            try:
                try:
                    difDF.iloc[ind]['FWHM'] = np.max([IdcmpH['FWHM'],TdcmpH['FWHM']])
                    difDF.iloc[ind]['ZPTMAG'] = IdcmpH['ZPTMAG']
                except Exception as e:
                    print('\n***ERROR:*** '+str(e))
#                    pass
            except Exception as e:
                print('\n***ERROR:*** '+str(e))
#                pass
        if difDF.iloc[ind]['ZPTMAG'] is np.nan:
            print('KLKMLKML')
            try:
                difDF.iloc[ind]['ZPTMAG'] = fishKW(difDF.iloc[ind]['Diff_HDU'].header,['ZPTMAG00','ZPTMAG','MAGZERO'])
            except Exception as e:
                pass
        w = wcs.WCS(difDF.iloc[ind]['Diff_HDU'].header)
        difDF.iloc[ind]['FWHM_ang'] = LEtb.pix2ang(w,difDF.iloc[ind]['FWHM'])
    return difDF


def LightCurves(LCs_file):
    tbl=asc.read(LCs_file)
    
    [DM15s, dm15_inv_inds] = np.unique(np.array(tbl["dm15"]), return_inverse=True)
    first_row = True
    for dm15_ind in np.arange(len(DM15s)):
        inds = np.argwhere(dm15_inv_inds==dm15_ind)
        [DELTAs, delta_inv_inds] = np.unique(np.array(tbl["delta"][inds]), return_inverse=True)
        
        for delta_ind in np.arange(len(DELTAs)):
            Dinds = np.argwhere(delta_inv_inds==delta_ind)
            
            dm15    = DM15s[dm15_ind]
            delta   = float(        tbl["delta"]    [inds[Dinds[0]]])
            psbnd   =               tbl["passband"] [inds[Dinds[0]]]
            phases  = SQ(np.array(  tbl["phase"]    [inds[Dinds]]))
            mags    = SQ(np.array(  tbl["mag"]      [inds[Dinds]]))            
            func_M, func_L = LCdata2func(phases,mags)

            if first_row:
                LC_tbl = Table({'dm15':     [dm15]  ,
                                'delta':    [delta] ,
                                'passband': [psbnd] ,
                                'phases':   [phases],
                                'mags':     [mags]  ,
                                'func_M':   [func_M],
                                'func_L':   [func_L]})
                first_row = False
            else:
                LC_tbl.add_row([dm15, delta, psbnd, phases, mags, func_M, func_L])
        
    return LC_tbl

def LCdata2func(phases,mags_p,smooth_lin=False,extrapolate=False):
    
    if extrapolate:
        phases, mags_p = extrap_mags(phases,mags_p)
    
    back_mag = np.Inf#np.max([mags_p[0], mags_p[-1]])
    back_LinFlux = 0.0
    LinFlux_p = DF.Mag2ADU(mags_p)
    phs_res = 2
    
    phases_shft = phases.copy() - phases[np.nanargmin(mags_p)]
#    phases_shft = phases.copy()# - 1
    def func_mag(phase):
        mag = np.interp(phase, phases_shft, mags_p, left=back_mag, right=back_mag)
        return mag
    
    if smooth_lin:
        min_phs = np.nanmin(phases_shft)
        max_phs = np.nanmax(phases_shft)
        
        LF_statistic1, phs_bin_edges1, _ = stats.binned_statistic(phases_shft,LinFlux_p,statistic='mean',bins=1+int((max_phs-min_phs)/phs_res))
        LF_statistic2, phs_bin_edges2, _ = stats.binned_statistic(phases_shft,LinFlux_p,statistic='mean',bins=1+int((max_phs-min_phs)/phs_res),range=(phs_bin_edges1[0]+np.mean(np.diff(phs_bin_edges1))/2,phs_bin_edges1[-1]))
#        LF_statistic1 = LF_statistic1/(LF_statistic2/np.nanmean(LF_statistic2))
        phs_bin_cntrs1 = (phs_bin_edges1[0:-1]+phs_bin_edges1[1:])/2
        inds=~np.isnan(LF_statistic1)
        LF_statistic1 = np.interp(phs_bin_cntrs1,phs_bin_cntrs1[inds],LF_statistic1[inds])
        phs_bin_cntrs2 = (phs_bin_edges2[0:-1]+phs_bin_edges2[1:])/2
        inds=~np.isnan(LF_statistic2)
        LF_statistic2 = np.interp(phs_bin_cntrs2,phs_bin_cntrs2[inds],LF_statistic2[inds])
        a=np.concatenate((phs_bin_cntrs1,phs_bin_cntrs2))
        b=np.concatenate((LF_statistic1,LF_statistic2))
        phs_bin_cntrs = np.linspace(a.min(),a.max(),a.size*10)#np.sort(a)#
        LF_statistic  = np.interp(phs_bin_cntrs,a,b)
        LF_statistic = (gaussian_filter(LF_statistic,sigma=(2/np.nanmean(np.diff(phs_bin_cntrs)))/2*np.sqrt(2*np.log(2))))
#        phs_diff = np.diff(phases_shft)
#        phases_upsamp = np.linspace(min_phs,max_phs,1+int((max_phs-min_phs)/np.nanmin(phs_diff[np.nonzero(phs_diff)])))
#        LinFlux_upsamp = np.interp(phases_upsamp,phases_shft,LinFlux_p)
#        LinFlux_smooth = gaussian_filter(LinFlux_upsamp, sigma = 40)
#        phs_bin_cntrs = phs_bin_cntrs - phs_bin_cntrs[np.nanargmax(LF_statistic)]
        LF_statistic = LF_statistic/np.nanmax(LF_statistic)
        def func_LinFlux(phase):
            LinFlux = np.interp(phase, phs_bin_cntrs, LF_statistic, left=back_LinFlux, right=back_LinFlux)
#            LinFlux = np.interp(phase, phases_upsamp, LinFlux_smooth, left=back_LinFlux, right=back_LinFlux)
            return LinFlux
    
    else:
        def func_LinFlux(phase,edge_query=None):
            if edge_query:
                return [phases_shft.min(), phases_shft.max()]
            else:
                LinFlux = np.interp(phase, phases_shft, LinFlux_p, left=back_LinFlux, right=back_LinFlux)
                return LinFlux
    
    return func_mag, func_LinFlux

def extrap_mags(phases,mags_p):
    # early_mag = mags_p[0]
    # late_mag = mags_p[-1]
    # mags_diff = np.diff(mags_p)
    
    # peak = mags_p.min()
    # mags_p_lst = mag_p.copy().tolist()
    
    return phases, mags_p

def fishKW(header,KWlist):
    KWval = None
    for KW in KWlist:
        try:
            KWval = header[KW]
        except:# Exception as e:
            pass
    return KWval

def dcmp2df(dcmp_file):
    hdr = fits.getheader(dcmp_file)
    cols=[]
    pattern = re.compile('COLTBL[0-9]{1,}')
    for k in hdr.keys():
        if pattern.match(k) is not None:
            cols.append(hdr[k])
    dat = fits.getdata(dcmp_file)
    str_dat = StringIO(' '.join(cols)+dat.tostring().decode('utf-8'))
    df = pd.read_csv(str_dat,sep='[ ]{1,}')
    return df
