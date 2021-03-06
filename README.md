#### **CHaracterization Algorithm for Radius Measurements (CHARM)**

**Authors:** UofA Group (F. Ozel, D. Psaltis, S. Dougall, T. Trent)

**Version:** 2.1

Date:October 13, 2021

Python scipt to characterize a black-hole image, measure its diameter and FWHM, as well as characterize its visibility map.

This version of the script does not assume a known center of the black-hole shadow but insted finds first the center of the image by minimizing an appropriate measure of the variance of diameters along different azimuthal cross sections.

This version is also meant to be applied to images reconstructed from real and synthetic data, i.e., without substantial substructure. For this reason the algorithm does not apply any filtering to the images.

The script runs in batch mode, by characterizing all images in files that end with '.fits' found in the directory given in variable 'path'

To execute, go to the /data directory and give:
     `python ../src/charm.py`

Upon completion, the script will generate a diagnostic plot for each of the images, as well as an ASCII file with name given in variable 'outfile' that containt, for each image, the following parameters:

    rad25 a float with the 25th percentile value of the radius
    rad50 a float with the 50th percentile value of the radius
    rad75 a float with the 75th percentile value of the radius
    FWHM25 a float with the 25th percentile value of the FWHM
    FWHM05 a float with the 50th percentile value of the FWHM
    FWHM75 a float with the 75th percentile value of the FWHM
    fracCirc a float with the fraction of the ring circunference with brightness above the floor
    zFirstHor the baseline length of the first horizontal visibility minimum in Glambda
    zFirstVer the baseline length of the first vertical visibility minimum in Glambda

Full documentation click [here](https://dpsaltis.github.io/CHARM/html)

