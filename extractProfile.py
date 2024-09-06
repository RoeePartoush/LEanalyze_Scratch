# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# astropy imports
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle, Latitude, Longitude
import astropy.units as u
# from astropy.io import ascii
# from astropy.time import Time
# from astropy import wcs
# from astropy.visualization import ZScaleInterval

# local modules
import File2Data as F2D
import DustFit as DFit
import LEtoolbox as LEtb
import LEplots_lite

def write_flux_profile(FP, out_file):
    """
    Write the flux profile to a file
    :param FP_df: DataFrame with the flux profile
    :param out_file: Path to the output file
    :return: None
    """
    x = FP['ProjAng']
    y = FP['FluxProfile_ADU']
    xy_df = pd.DataFrame({'x': x, 'y': y})
    xy_df.to_csv(out_file, index=False)
    print(f"Flux profile saved to {out_file}")

def main():
    DEBUG = False
    # ========== SET SLIT PARAMETERS ==========
    # Create the parser
    parser = argparse.ArgumentParser(description="A script to demonstrate argparse usage")

    # Add arguments
    parser.add_argument('fits_file', type=str, help="Path to the input file")
    parser.add_argument('RA', type=float, help="Flux profile center RA (in degrees)")
    parser.add_argument('DEC', type=float, help="Flux profile center DEC (in degrees)")
    parser.add_argument('PA', type=float, help="Flux profile position angle (in degrees)")
    parser.add_argument('L', type=float, help="Flux profile length (in arcsec)")
    parser.add_argument('W', type=float, help="Flux profile width (in arcsec)")
    parser.add_argument('--out_prof', type=str, default=None, help="Path to the output flux profile file (optional)")
    parser.add_argument('--out_img', type=str, default=None, help="Path to the output file (optional)")

    # Parse the arguments
    args = parser.parse_args()
    if DEBUG:
        Orgs=SkyCoord([(12.37191576, 58.76229830)],frame='fk5',unit=(u.deg, u.deg))
        PA = Angle([Angle(150,'deg') for Org in Orgs])+Angle(180,u.deg)
        Ln = Angle([40  for Org in Orgs],u.arcsec)
        Wd = Angle([2  for Org in Orgs],u.arcsec)
    else:
        Orgs=SkyCoord([(args.RA, args.DEC)],frame='fk5',unit=(u.deg, u.deg))
        PA = Angle([Angle(args.PA,'deg') for Org in Orgs])+Angle(180,u.deg)
        Ln = Angle([args.L  for Org in Orgs],u.arcsec)
        Wd = Angle([args.W  for Org in Orgs],u.arcsec)

    # LOAD DATA

    if DEBUG:
        # fits_path = "/Users/roeepartoush/Downloads/F150W_sw_i2d"
        # fits_path = "/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/KECK/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd"
        fits_path = ".data/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd"
    else:
        fits_path = args.fits_file
        if not os.path.exists(fits_path):
            raise FileNotFoundError(f"File not found: {fits_path}")

    files = [fits_path]
    DIFF_df = F2D.FitsDiff(files)

    clmns = ['Orig', 'PA', 'Length','WIDTH']
    slitFPdf = pd.DataFrame(index=np.arange(len(Orgs)), columns=clmns, data = [(Orgs[i],PA[i],Ln[i],Wd[i]) for i in np.arange(len(Orgs))])

    # ========== EXTRACT PROFILE FROM IMAGE ==========
    print('\n\n=== Extracting Flux Profiles... ===')
    FP_df_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf, REF_image=None, N_bins=int(Ln[0].arcsec),uniform_wcs=False)

    # ========== PLOT IMAGES & PROFILES ==========
    print('\n\n=== Plotting Images... ===')
    plt.close('all')
    figures=[plt.figure()]# for i in DIFF_df.index]
    axs = LEplots_lite.imshows(DIFF_df,plot_Fprofile=True, profDF=slitFPdf, prof_sampDF_lst=FP_df_lst, FullScreen=False, figs=figures)
    w_s = DIFF_df['WCS_w'].to_list()
    LEplots_lite.match_zoom_wcs(axs,w_s,slitFPdf.iloc[0]['Orig'],slitFPdf.iloc[0]['Length']*1.50,slitFPdf.iloc[0]['Length']*1.5)
    if DEBUG:
        plt.show()
    
    # ========== SAVE IMAGES & PROFILES ==========
    if args.out_img is not None:
        figures[0].savefig(args.out_img)
        print(f"Image saved to {args.out_img}")
    if args.out_prof is not None:
        write_flux_profile(FP_df_lst[0].iloc[0], args.out_prof)

if __name__ == "__main__":
    main()