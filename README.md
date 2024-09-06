# README

To run `extractProfile.py`, use the following command line:

```
python extractProfile.py some_image.fits 12.37 58.76 150 40 2 --out_prof profile.csv --out_img plot.png
```
Positional args order is as follows:
input fits file path, RA, DEC, profile PA, profile length, profile width.

Then, ```--out_prof``` and ```--out_img``` are paths for the flux profile data and a figure ploting the profile over the image, respectively.
