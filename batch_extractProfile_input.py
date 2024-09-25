#!/Users/jjencson/anaconda3/envs/stenv/bin/python

import os
import numpy as np
import argparse
from astropy.coordinates import SkyCoord
import astropy.units as u

def main():
    parser = argparse.ArgumentParser(description="A script to generate an grid of parallel profile boxes and an input script to batch run extractProfile.py")
    
    parser.add_argument('image', type=str, help="path to image where you want to run extractions.")
    parser.add_argument('RA', type=float, help="RA of center of grid")
    parser.add_argument('DEC', type=float, help='Dec of center of profile box grid')
    parser.add_argument('PA', type=float, help='position angle of profile boxes in degrees')
    parser.add_argument('L', type=float, help="length of each profile boxes in arcsec")
    parser.add_argument('W', type=float, help='width of profile boxes in arcsec')
    parser.add_argument('nW', type=int, help='number of boxes to generate along width axis')
    parser.add_argument('--nL', type=int, help='number of boxes to generate along length axis', default=1)
    parser.add_argument('--outdir', type=str, help='where to put the output', default='')
    parser.add_argument('--outfile', type=str, help='Name of the output file, otherwise default to long named based on input parameters.', default=None)
    parser.add_argument('--plot', action='store_true', help='generate plots for each profile box (optional)')
    parser.add_argument('--regions', action='store_true', help="make a region file of the boxes.")
    parser.add_argument('--reg_color', type=str, default='cyan', help="color for ds9 region file.")
    parser.add_argument('--align_W', action='store_true', help='align boxes along PA in width axis, instead of along RA/Dec')
    parser.add_argument('--wcs_ext', type=int, default=0, help="index of extension in the fits file containing the wcs (default is 0).")
    
    args = parser.parse_args()

    image = args.image
    RA = args.RA * u.deg
    DEC = args.DEC * u.deg
    PA = args.PA * u.deg
    L = args.L * u.arcsec
    W = args.W  * u.arcsec
    nW = args.nW
    nL = args.nL
    outdir = args.outdir
    outfile = args.outfile
    plot = args.plot
    regions = args.regions
    reg_color = args.reg_color
    align_W = args.align_W
    wcs_ext = args.wcs_ext
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    if outfile is None:
        outfile = f"batch_{os.path.basename(image).split('.fits')[0]}_RA{RA.value:0.7f}_DEC{DEC.value:0.7f}_PA{PA.value:0.1f}_L{L.value:0.1f}_W{W.value:0.3f}_{nL}x{nW}"
    outf = open(os.path.join(outdir, outfile), 'w')
    
    coord0 = SkyCoord(RA, DEC, frame='icrs')
    
    if regions:
        out_reg = outfile+'.reg'
        regf = open(os.path.join(outdir, out_reg), 'w')
        regf.write(f"""# Region file format: DS9 version 4.1
global color={reg_color} dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
icrs
""")
        
    #numbers need to generate the grid
    nW_inthalf = int(nW/2)
    nL_inthalf = int(nL/2)
    
    if align_W: 
        #offset by W in rotated PA frame
        Wscale = 1.0
        Wangle = 90.0*u.deg+PA
    else: 
        #offset by W in RA/Dec frame
        Wscale = 1.0/np.cos(PA.to(u.rad)) #not perfect, not sure if I need some spherical geometry term or something... 
        Wangle = 90.0*u.deg        

    for i in range(-1*nL_inthalf, -1*nL_inthalf + nL, 1):
        for j in range(-1*nW_inthalf, -1*nW_inthalf + nW, 1):
            box_coord = coord0.directional_offset_by(Wangle, W*Wscale*j)
            box_coord = box_coord.directional_offset_by(PA, L*i) #length offset, aligned along PA
            
            out_prof = os.path.join(outdir, f"{os.path.basename(image).split('.fits')[0]}_RA{box_coord.ra.deg:0.7f}_DEC{box_coord.dec.deg:0.7f}_PA{PA.value:0.1f}_L{L.value:0.1f}_W{W.value:0.3f}.csv")
            if plot:
                plot_str = f" --out_img {out_prof.replace('csv', 'png')}"
            else:
                plot_str = ''
            
            outf.write(f"python extractProfile.py {image} {box_coord.ra.deg:0.7f} {box_coord.dec.deg:0.7f} {PA.value:0.2f} {L.value:0.3f} {W.value:0.3f} --out_prof {out_prof} {plot_str} --wcs_ext {wcs_ext} \n")
            
            if regions: 
                regf.write(f"box({box_coord.ra.deg:0.7f},{box_coord.dec.deg:0.7f},{W.value:0.2f}\",{L.value:0.2f}\",{PA.value:0.2f})\n")
    
    if regions:
        regf.close()
    
    outf.close()

if __name__ == "__main__":
    main()