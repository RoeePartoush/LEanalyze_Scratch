# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import scipy as sp
from sklearn.mixture import GaussianMixture

# import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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

# def fit_GMM_1d(FP, N_g):
    # x = FP['ProjAng'].arcsec
    # y = FP['FluxProfile_ADU']
#     model = GaussianMixture(N_g,)
#     return

# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Define the sum of multiple Gaussians
def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude = params[i]
        mean = params[i + 1]
        stddev = params[i + 2]
        y += gaussian(x, amplitude, mean, stddev)
    return y

# Function to fit the signal to the model
def fit_GMM_1d(x, y, num_gaussians, initial_params=None):
    # Generate initial parameters if not provided
    if initial_params is None:
        initial_params = []
        for _ in range(num_gaussians):
            initial_params.extend([max(y), np.mean(x), np.std(x)])
            
    # Fit the signal
    params, covariance = curve_fit(multi_gaussian, x, y, p0=initial_params, maxfev=20000)
    
    return params

# # Example usage
# # Create synthetic data for demonstration
# np.random.seed(0)
# x_data = np.linspace(-10, 10, 200)
# y_data = (gaussian(x_data, 3, -2, 1) + 
#           gaussian(x_data, 5, 0, 1.5) + 
#           gaussian(x_data, 2, 3, 0.5) + 
#           0.5 * np.random.normal(size=x_data.size)) # Adding noise

# # Fit the data to 3 Gaussians
# num_gaussians = 3
# params = fit_signal(x_data, y_data, num_gaussians)

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(x_data, y_data, 'b.', label='Data')
# plt.plot(x_data, multi_gaussian(x_data, *params), 'r-', label='Fitted Model')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('1D Signal Fitting with Gaussian Mixture Model')
# plt.show()

# Print the fitted parameters
# print("Fitted parameters:")
# for i in range(num_gaussians):
#     amplitude, mean, stddev = params[3 * i], params[3 * i + 1], params[3 * i + 2]
#     print(f"Gaussian {i+1}: Amplitude = {amplitude:.2f}, Mean = {mean:.2f}, Stddev = {stddev:.2f}")


def write_flux_profile(FP, out_file):
    """
    Write the flux profile to a file
    :param FP_df: DataFrame with the flux profile
    :param out_file: Path to the output file
    :return: None
    """
    x = Angle(FP['ProjAng']).arcsec
    y = FP['FluxProfile_ADU']
    xy_df = pd.DataFrame({'arcsec': x, 'Flux': y})
    xy_df.to_csv(out_file, index=False)
    print(f"Flux profile saved to {out_file}")

def main():
    DEBUG = True
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
    parser.add_argument('--fit_gmm', type=int, default=None, help="If present, an integer specifiying number of gaussians for profile fit.")
    parser.add_argument('--wcs_ext', type=int, default=0, help="index of extension in the fits file containing the wcs (default is 0).")

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
        fits_path = "./example/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd.fits"
    else:
        fits_path = args.fits_file
        if not os.path.exists(fits_path):
            raise FileNotFoundError(f"File not found: {fits_path}")

    files = [fits_path]
    DIFF_df = F2D.FitsDiff(files, args.wcs_ext)

    clmns = ['Orig', 'PA', 'Length','WIDTH']
    slitFPdf = pd.DataFrame(index=np.arange(len(Orgs)), columns=clmns, data = [(Orgs[i],PA[i],Ln[i],Wd[i]) for i in np.arange(len(Orgs))])

    # ========== EXTRACT PROFILE FROM IMAGE ==========
    print('\n\n=== Extracting Flux Profiles... ===')
    FP_df_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf, REF_image=None, N_bins=int(Ln[0].arcsec),uniform_wcs=False)

    if args.fit_gmm:
        # ========== FIT PROFILE WITH GMM ==========
        print('\n\n=== Extracting Flux Profiles... ===')
        FP = FP_df_lst[0]
        x_data = Angle(FP['ProjAng']).arcsec.flatten()
        y_data = FP['FluxProfile_ADU'][0]
        params = fit_GMM_1d(x_data, y_data, args.fit_gmm, initial_params=None)

    # ========== PLOT IMAGES & PROFILES ==========
    print('\n\n=== Plotting Images... ===')
    plt.close('all')
    figures=[plt.figure()]
    axs_img, axs_flux = LEplots_lite.imshows(DIFF_df,plot_Fprofile=True, profDF=slitFPdf, prof_sampDF_lst=FP_df_lst, FullScreen=False, figs=figures)
    w_s = DIFF_df['WCS_w'].to_list()
    LEplots_lite.match_zoom_wcs(axs_img,w_s,slitFPdf.iloc[0]['Orig'],slitFPdf.iloc[0]['Length']*1.50,slitFPdf.iloc[0]['Length']*1.5)

    if args.fit_gmm:
        # plot profile fit
        axs_flux[0].plot(x_data, y_data, 'b.', label='Data')
        axs_flux[0].plot(x_data, multi_gaussian(x_data, *params), 'r-', label='Fitted Model')
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