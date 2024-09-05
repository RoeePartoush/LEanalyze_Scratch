# -*- coding: utf-8 -*-


# astropy imports
from astropy.coordinates import SkyCoord  # High-level coordinates
#from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy.io import ascii
from astropy.time import Time
from astropy import wcs
from astropy.visualization import ZScaleInterval

# local modules
import File2Data as F2D
import DustFit as DFit
import LEtoolbox as LEtb
import LEplots


# ========= USER INPUT =========
# == COPY AND PASTE OVER HERE ==
# ------------------------------
Orgs=SkyCoord([(12.37191576, 58.76229830)],frame='fk5',unit=(u.deg, u.deg))
PA = Angle([Angle(150,'deg') for Org in Orgs])+Angle(180,u.deg)
Ln = Angle([40  for Org in Orgs],u.arcsec)
Wd = Angle([2  for Org in Orgs],u.arcsec)

# LOAD DATA
fits_path = "/Users/roeepartoush/Downloads/F150W_sw_i2d"
files = [fits_path]
DIFF_df = F2D.FitsDiff(files)
# %%

clmns = ['Orig', 'PA', 'Length','WIDTH']
slitFPdf = pd.DataFrame(index=np.arange(len(Orgs)), columns=clmns, data = [(Orgs[i],PA[i],Ln[i],Wd[i]) for i in np.arange(len(Orgs))])

# ========== EXTRACT PROFILE FROM IMAGE ==========
print('\n\n=== Extracting Flux Profiles... ===')
FP_df_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf, REF_image=None, N_bins=int(Ln[0].arcsec),uniform_wcs=False)