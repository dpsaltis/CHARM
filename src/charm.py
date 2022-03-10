## @package charm
#
# Python scipt to characterize a black-hole image
# This version finds first the center of the image
# and does not invoke any filtering
#
#    \author UofA Group
#  
#    \version 2.1
#  
#    \date October 19, 2021
#
#    \bug No known bugs
#    
#    \warning No known warnings
#    
#    \todo Nothing left

## @mainpage CHaracterization Algorithm for Radius Measurements (CHARM)
#
# Python scipt to characterize a black-hole image, measure its diameter
# and FWHM, as well as characterize its visibility map.
# 
# This version of the script does not assume a known center of the black-hole
# shadow but insted finds first the center of the image by minimizing an
# appropriate measure of the variance of diameters along different azimuthal
# cross sections.
#
# This version is also meant to be applied to images reconstructed from
# real and synthetic data, i.e., without substantial substructure. For this
# reason the algorithm does not apply any filtering to the images.
#
# The script runs in batch mode, by characterizing all images in files that
# end with '.fits' found in the directory given in variable 'path'
#
# To execute, go to the /data directory and give:
#      python ../src/charm.py
#
# Upon completion, the script will generate a diagnostic plot for each of
# the images, as well as an ASCII file with name given in variable 'outfile'
# that containt, for each image, the following parameters:
#
#     rad25 a float with the 25th percentile value of the radius
#     rad50 a float with the 50th percentile value of the radius
#     rad75 a float with the 75th percentile value of the radius
#     FWHM25 a float with the 25th percentile value of the FWHM
#     FWHM05 a float with the 50th percentile value of the FWHM
#     FWHM75 a float with the 75th percentile value of the FWHM
#     fracCirc a float with the fraction of the ring circunference with brightness above the floor
#     zFirstHor the baseline length of the first horizontal visibility minimum in Glambda
#     zFirstVer the baseline length of the first vertical visibility minimum in Glambda
#    \author UofA Group (F. Ozel, D. Psaltis, S. Dougall, T. Trent)
#  
#    \version 2.1
#  
#    \date October 13, 2021
#
#    \bug No known bugs
#    
#    \warning No known warnings
#    
#    \todo Nothing left

# necessary libraries
import sys
import os

import numpy as np                    # imports library for math

import matplotlib as mpl
mpl.use('Agg')                        # use it in batch mode
import matplotlib.pyplot as plt       # imports library for plots

from matplotlib import rcParams       # import to change plot parameters
import plotParams                     # to set up the figure parameters

import scipy.special as sp            # import scipy
import scipy.interpolate as interpolate # import scipy interpolate
from matplotlib import cm             # imports colormaps

from astropy.io import fits

# Global Parameters
d0X=0.0                               # X displacement of center of image
d0Y=0.0                               # Y displacement of center of image

convert=1./206.26                     # microarcsec to Glambda

RadMax=50.0                           # max radial size of image
Ngrid=256                             # number of radial grid points for slices
Nslice=128                            # number of azimuthal slices
iFloor=0.10                           # percent of max image brightness considered a floor

# the value of iFloor depends on the dynamical range of images; it can be very
# small for theoretical images, and as large as 0.1 for reconstructed images
# given the factor of 10 dynamical range of current EHT images

def zcr(x, y):
    """!@brief Finds zero crossings
    
    Given two arrays of the x- and y- coordinates of a function
    it returns an array of the x-coodinates of the zero crossings
    
    the x-coordinates are assumed to be in order

    @param x a float array with the x-coodinates
    @param y a float array with the y-coordinates

    @returns a float array with the locations of the zero crossings

    \author UofA Group
  
    \version 1.0
  
    \date September 13, 2018

    \bug No known bugs
    
    \warning No known warnings
    
    \todo Nothing left
    """
    return x[np.diff(np.sign(y),append=1) != 0]


def findcenter(Npixels,X,Y,ImageFilter,Dmax=25.,ND=50,Ngrid=Ngrid,Nslice=Nslice,iFloor=iFloor,radmin=15):
    """!@brief # Finds the center of the image by minimizing the
    spread of the radii measured along different azimuthal slices

    Given a square image of Npixels per side, with coordinates in X and Y
    and brightness in ImageFilter, it finds its center by searching 
    through a square grid of NDxND points in the range [-Dmax,Dmax] 
    along each orientation. The two parameters are inputs but defaults 
    are Dmax=25 and ND=50.

    The remaining default parameters are from the global variables.

    In this version of the script, there is no image filtering; any
    mention of fitering is for compatibility with other script
    versions.

    All image coordinates and distances are in the native units of the
    image provided.
  
    v2.0 There is now a minimum radius requirement for the ring stored
    in 'radmin'. Introducing this ensures that the algorithm doesn't
    focus on compact knots. This value can be different between the
    centering and the characterization function, with the former
    typically being larger.

    @param Npixels the number of points per dimension (assuming square image)
    @param X a 1D array of Npts points with the X-coordinates
    @param Y a 1D array of Npts points with the Y-coordinates
    @param ImageFilter a 2D array of Npts*Npts points with the image brightness

    @returns d0X a float with the x-displacement of the center
    @returns d0Y a float with the y-displacement of the center

    \author UofA Group
  
    \version 2.0
  
    \date October 19, 2021

    \bug No known bugs
    
    \warning No known warnings
    
    \todo Increase speed by using segmented search

    """
    # find the max brightness of the image
    ifilterMax=np.amax(ImageFilter)
    # interpolate the 2D image
    ImageInterp=interpolate.RectBivariateSpline(X, Y, ImageFilter)

    # find max X and Y grid sizes of image, assumign square grid
    Xmax=np.max(X)
    Ymax=np.max(Y)

    # make a grid of Nslice points in azimuthal slices
    sliceangle=np.linspace(0.,2.*np.pi,Nslice)

    # aux variables to store displacement of current best guess
    # for image center
    d0Xmin=0.
    d0Ymin=0.
    rmsmin=1.e10    # something very large

    # now search for the location of the center
    for d0X in np.linspace(-Dmax,Dmax,ND):
        for d0Y in np.linspace(-Dmax,Dmax,ND):

            # make a radial grid for each slice with Ngrid points up to edge
            # which ever orientation it is
            RadEdge=np.min([Xmax-d0X,Ymax-d0Y])
            distance=np.linspace(0.,RadEdge,Ngrid)
            # this is the stepsize in radius
            stepRadius=RadEdge/(Ngrid-1.0)

            # make an array of zeros to store the radii for each azimuthal slice
            radii=np.zeros(Nslice)
            
            for index in np.arange(Nslice):
                angle=sliceangle[index]
                # make two 1D arrays with X- and Y- coordinates along the slice
                # the X-coordinate is displaced by d0X and the y by d0Y 
                Xslice=distance*np.cos(angle)+d0X
                Yslice=distance*np.sin(angle)+d0Y
                # interpolate the image along this slice
                ImageSlice=ImageInterp.ev(Xslice,Yslice)
                # max intensity of this slice
                ISliceMax=np.max(ImageSlice)
                
                # only consider this slice if its peak brightness is above the floor value
                if (ISliceMax>=iFloor*ifilterMax):
                    
                    # find the index of the maximum along the slice
                    iMax = np.where(ImageSlice == np.amax(ImageSlice))
                    radii[index]=distance[iMax[0][0]]
                else:
                    radii[index]=0.0

            # only continue if the radii of max brightness are larger
            # than radmin (to avoid centering on a bright blob)
            # for at least half the slices
            if (np.size(radii[radii>radmin])>Nslice/2):
                #calculate the 85th-15th percentile variance of the radii along the different slices
                rms=np.percentile(radii,85)-np.percentile(radii,15)
                # if this is the smallest so far, keep it as a good guess
                if ((rms<rmsmin) & (np.median(radii)>radmin)):
                    rmsmin=rms
                    d0Xmin=d0X
                    d0Ymin=d0Y

    return d0Xmin, d0Ymin
            

def FilterImage(Npts,X,Y,I,dX,Npadfact=16):
    """!@brief (In principle) filters an image and returns its visibility map

    For this version of the script, there is no filtering.

    Given a square image of Npts per side, with coordinates in X and Y
    and brightness in I, the subroutine just returns the visibility map 
    of the image.

    Npadfact is the multiplicative padding factor for calculating the 2D
    Fourier transform of the image with default Npadfact=16

    @param Npts the number of points per dimension (assuming square image)
    @param X a 1D array of Npts points with the X-coordinates
    @param Y a 1D array of Npts points with the Y-coordinates
    @param I a 2D array of Npts*Npts points with the image brightness
    @param dX a float with the pixel width in uas
    
    @returns FilterImage a 2D Npts*Npts array with the filtered image
    @returns uGrid a 2D Npts*Npts array with the u-coordinate grid in Glambda
    @returns vGrid a 2D Npts*Npts array with the v-coordinate grid in Glambda
    @returns Visibility a 2D Npts*Npts array with the complex Visibility

    \author UofA Group
  
    \version 1.0
  
    \date September 13, 2018

    \bug No known bugs
    
    \warning No known warnings
    
    \todo Nothing left
    """

    # padding
    Npad=Npadfact*Npts
    # take the fourier transform
    Visibility=np.fft.fftn(I,s=[Npad,Npad])
    # make a u-v grid
    uCo=np.fft.fftfreq(Npad,dX)/convert
    vCo=np.fft.fftfreq(Npad,dX)/convert
    
    # set up the filter
    uGrid, vGrid = np.meshgrid(uCo, vCo)
    gmask = 1     # no filter for the reconstructed images!
    
    # apply the filter mask
    VisibilityFilter = Visibility * gmask
    
    # go back to the image plane (and use only the magnitude of the complex FFT)
    ImageFilter=np.abs(np.fft.ifftn(VisibilityFilter))
    # remove padding 
    ImageFilter=ImageFilter[:Npts,:Npts]

    return ImageFilter,uGrid,vGrid,Visibility


def ImageParams(Npixels,X,Y,ImageFilter,d0X=d0X,d0Y=d0Y,Ngrid=Ngrid,Nslice=Nslice,iFloor=iFloor,radmin=5.):
    """!@brief Measures the diameter and FWHM of a ring-like image

    Given a square image of Npixels per side, with coordinates in X and Y,
    brightness in ImageFilter, and image center at d0X and d0Y, 
    measures the distribution of "radii" and "widths" of the image
    for a grid of Nslice azimuthal slices with a distance up to RadMax

    It only considers slices for which the max brightness is larger than
    iFloor*ifilterMax, which is important for, e.g., CLEAN images that have
    a substantial central floor.

    v2.0 There is now a minimum radius requirement for the ring stored
    in 'radmin'. Introducing this ensures that the algorithm doesn't
    focus on compact knots. This value can be different between the
    centering and the characterization function, with the former
    typically being larger.

    @param Npixels the number of points per dimension (assuming square image)
    @param X a 1D array of Npts points with the X-coordinates
    @param Y a 1D array of Npts points with the Y-coordinates
    @param ImageFilter a 2D array of Npts*Npts points with the image brightness
    @param d0X a float with the x-displacement of the center
    @param d0Y a float with the y-displacement of the center

    @returns sliceangle a 1D array with the angles (from horizontal) of the slices 
    @returns radii a 1D array with the peak brightness radii of the individual slices
    @returns radiusL a 1D array with the radius of the half point on the left
    @returns radiusR a 1D array with the radius of the half point on the right

    \author UofA Group
  
    \version 2.0
  
    \date October 19, 2021

    \bug No known bugs
    
    \warning No known warnings
    
    \todo Nothing left
    """

    # find the max of the image
    ifilterMax=np.amax(ImageFilter)
    # interpolate the 2D image
    ImageInterp=interpolate.RectBivariateSpline(X, Y, ImageFilter)
    # make a radial grid for each slice with Ngrid points
    distance=np.linspace(0.,RadMax,Ngrid)
    # this is the stepsize in radius
    stepRadius=RadMax/(Ngrid-1.0)
    # make a grid of Nslice points in azimuthal slices
    sliceangle=np.linspace(0.,2.*np.pi,Nslice)
    # make a arrays of zeros to store the radii for each azimuthal slice
    radii=np.zeros(Nslice)
    radiusL=np.zeros(Nslice)
    radiusR=np.zeros(Nslice)

    for index in np.arange(Nslice):
        angle=sliceangle[index]
        # make two 1D arrays with X- and Y- coordinates along the slice
        # the X-coordinate is displaced by dX0, Y-coordinate by d0Y 
        Xslice=distance*np.cos(angle)+d0X
        Yslice=distance*np.sin(angle)+d0Y
        # interpolate the image along this slice
        ImageSlice=ImageInterp.ev(Xslice,Yslice)
        # max intensity of this slice
        ISliceMax=np.max(ImageSlice)
        
        # only consider this slice if its peak brightness is above the floor value
        if (ISliceMax>=iFloor*ifilterMax):
            
            # find the index of the maximum along the slice
            iMax = np.where(ImageSlice == np.amax(ImageSlice))
            radii[index]=distance[iMax[0][0]]

            # for DEBUG only
            #for ind1 in np.arange(Ngrid):
            #    print(angle,distance[ind1],ImageSlice[ind1])

            # find the regions to the left and right of max
            ImageLeft=ImageSlice[:iMax[0][0]+1]
            ImageRight=ImageSlice[iMax[0][0]:]

            # integral of brightness to the left of the peak
            intLeft=np.sum(ImageLeft)*stepRadius/ISliceMax
            # equivalent width
            radiusL[index]=radii[index]-2.*np.sqrt(np.log(2)/np.pi)*intLeft
            if (radiusL[index]<0):
                radiusL[index]=0
            
            # integral to the right of the peak
            intRight=np.sum(ImageRight)*stepRadius/ISliceMax
            radiusR[index]=radii[index]+2.*np.sqrt(np.log(2)/np.pi)*intRight
            
    return sliceangle[radii>radmin],radii[radii>radmin],radiusL[radii>radmin],radiusR[radii>radmin]

################################################################
# function char_image(label,movieframe,Npixels, X,Y,I)
#
# main function to characterize an image of Npixels by Npixels,
# stored in the 2D array I, with horizontal and vertical coordinates
# stored in arrays X and Y. 
################################################################


def char_image(label,movieframe,Npixels, X,Y,I):
    """!@brief Image characterization subroutine

    Given a square image of Npixels per side, with coordinates in X and Y,
    and brightness in I, this subroutine first filters the image (not in this
    script), find its center, measures the distribution of "radii" and "widths" 
    for a grid of Nslice azimuthal slices, and plots all of the above together
    with two cross section of the visibility map of the image.
    
    Besides returning a lot of these parameters, it also generates
    a 6-panel diagnostic plot, labeled with the input stirng 'label'.

    The plot is saved as a PNG file with name movieframe+".png"
    If 'movieframe' corresponds to an increasing integer, the resulting
    files can be easily combined into a movie with ffmpeg.
    
    @label a string to be placed on the image, for recongition
    @movieframe a string for the name of the plot file
    @param Npixels the number of points per dimension (assuming square image)
    @param X a 1D array of Npts points with the X-coordinates
    @param Y a 1D array of Npts points with the Y-coordinates
    @param I a 2D array of Npts*Npts points with the image brightness


    @returns rad25 a float with the 25th percentile value of the radius
    @returns rad50 a float with the 50th percentile value of the radius
    @returns rad75 a float with the 75th percentile value of the radius
    @returns FWHM25 a float with the 25th percentile value of the FWHM
    @returns FWHM05 a float with the 50th percentile value of the FWHM
    @returns FWHM75 a float with the 75th percentile value of the FWHM
    @returns fracCirc a float with the fraction of pi with brightness above the floor
    @returns zFirstHor the baseline length of the first horizontal visibility minimum in Glambda
    @returns zFirstVer the baseline length of the first vertical visibility minimum in Glambda

    @returns nothing

    \author UofA Group
  
    \version 2.1
  
    \date October 22, 2021

    \bug No known bugs
    
    \warning No known warnings
    
    \todo Nothing left
    """
    # assuming a square, cartesian image, this is the FOV in uas
    FOV=(np.amax(X)-np.amin(X))
    # and this is pixel size in uas
    dX=FOV/Npixels

    # apply the Butterworth filter and get FFTs (no filtering in reconstructed images)
    ImageFilter,uCo,vCo,Visibility=FilterImage(Npixels,X,Y,I,dX)

    # find max brightness of filtered image
    ifilterMax=np.max(ImageFilter)

    # normalize image and filtered image
    I=I/ifilterMax
    ImageFilter=ImageFilter/ifilterMax
    ifilterMax=1.0

    # measure the radii and widths along slices
    sliceangle,radii,radiusL,radiusR=ImageParams(Npixels,X,Y,ImageFilter,d0X,d0Y,Ngrid=Ngrid,Nslice=Nslice,iFloor=iFloor)

    # calculate percentiles for the radii
    rad25=np.percentile(radii,25)
    rad50=np.percentile(radii,50)
    rad75=np.percentile(radii,75)
    # calculate FWHM
    FWHM=radiusR-radiusL
    FWHM25=np.percentile(FWHM,25)
    FWHM50=np.percentile(FWHM,50)
    FWHM75=np.percentile(FWHM,75)

    # calculate fraction of circumference that is "visible"
    fracCirc=np.size(sliceangle)/(1.0*Nslice)
    
    #### Graph results

    # setup the 6 panels
    figsize=(12,7.5) #size of the figure for general figures

    fig, ((ax1, ax2, ax3),(ax4,ax5,ax6)) = plt.subplots(ncols=3, nrows=2, figsize=figsize)

    # plot the unfiltered image on the left panel
    ax1.set(xlim=(np.amin(X),np.amax(X)),ylim=(np.amin(Y),np.amax(Y)))
    pos1=ax1.contourf(X,Y,np.abs(I.T),128,cmap=cm.gist_heat,vmax=1)
    ax1.set_xlabel(r'$x$ ($M$)')
    ax1.set_ylabel(r'$y$ ($M$)')
    
    # identify the run
    textlabel=label
    ax1.text(-60,60,textlabel,fontsize='small',color='white')
    
    # plot the cross sections on the second panel
    ax2.axis([np.amin(X),np.amax(X),0,1.3])

    # interpolate the 2D image
    ImageInterp=interpolate.RectBivariateSpline(X, Y, ImageFilter)
    # make a radial grid for each slice with Ngrid points
    distance=np.linspace(0.,RadMax,Ngrid)

    # make two 1D arrays with X- and Y- coordinates along the various slices
    Xslice=distance*np.cos(0)+d0X
    Yslice=distance*np.sin(0)+d0Y
    # interpolate the image along this slice
    ImageSlice=ImageInterp.ev(Xslice,Yslice)
    IsliceMaxXp=np.max(ImageSlice)
    ax2.plot(Xslice,ImageSlice,'r--',lw=0.5,label=r'Horizontal')
    Xslice=distance*np.cos(np.pi)+d0X
    Yslice=distance*np.sin(np.pi)+d0Y
    # interpolate the image along this slice
    ImageSlice=ImageInterp.ev(Xslice,Yslice)
    IsliceMaxXm=np.max(ImageSlice)
    ax2.plot(Xslice,ImageSlice,'r--',lw=0.5)

    # make two 1D arrays with X- and Y- coordinates along the various slices
    Xslice=distance*np.cos(np.pi/2.)+d0X
    Yslice=distance*np.sin(np.pi/2.)+d0Y
    # interpolate the image along this slice
    ImageSlice=ImageInterp.ev(Xslice,Yslice)
    IsliceMaxYp=np.max(ImageSlice)
    ax2.plot(Yslice,ImageSlice,'b--',lw=0.5,label=r'Vertical')
    Xslice=distance*np.cos(3.*np.pi/2.)+d0X
    Yslice=distance*np.sin(3.*np.pi/2.)+d0Y
    # interpolate the image along this slice
    ImageSlice=ImageInterp.ev(Xslice,Yslice)
    IsliceMaxYm=np.max(ImageSlice)
    ax2.plot(Yslice,ImageSlice,'b--',lw=0.5)
    
    ax2.set_xlabel(r'Angular Distance ($M$)')
    ax2.set_ylabel(r'Image')
    
    # plot the equivalent Gaussians as dotted lines
    # first positive X direction
    indwhere=np.argmax(np.cos(sliceangle))
    if (np.cos(sliceangle[indwhere])>0.95):
        sigmaL=-(radiusL[indwhere]-radii[indwhere])/np.sqrt(2.*np.log(2))
        Gaussian=IsliceMaxXp*np.exp(-(X-radii[indwhere])*(X-radii[indwhere])/2./sigmaL/sigmaL)
        condition=((X>0) & (X<radii[indwhere]))
        ax2.plot(X[condition]+d0X,Gaussian[condition],'m-',lw=1)
        sigmaR=(radiusR[indwhere]-radii[indwhere])/np.sqrt(2.*np.log(2))
        Gaussian=IsliceMaxXp*np.exp(-(X-radii[indwhere])*(X-radii[indwhere])/2./sigmaR/sigmaR)
        condition=((X>0) & (X>radii[indwhere]))
        ax2.plot(X[condition]+d0X,Gaussian[condition],'m-',lw=1)
        
    # then the negative X direction
    indwhere=np.argmin(np.cos(sliceangle))
    if (np.cos(sliceangle[indwhere])<-0.95):
        ISliceMax=np.amax(ImageFilter[:np.int(Npixels/2),np.int(Npixels/2)])
        sigmaL=-(radiusL[indwhere]-radii[indwhere])/np.sqrt(2.*np.log(2))
        Gaussian=IsliceMaxXm*np.exp(-(X+radii[indwhere])*(X+radii[indwhere])/2./sigmaL/sigmaL)
        condition=((X<0) & (np.abs(X)<radii[indwhere]))
        ax2.plot(X[condition]+d0X,Gaussian[condition],'m-',lw=1)
        sigmaR=(radiusR[indwhere]-radii[indwhere])/np.sqrt(2.*np.log(2))
        Gaussian=IsliceMaxXm*np.exp(-(X+radii[indwhere])*(X+radii[indwhere])/2./sigmaR/sigmaR)
        condition=((X<0) & (np.abs(X)>radii[indwhere]))
        ax2.plot(X[condition]+d0X,Gaussian[condition],'m-',lw=1)

    # then the positive Y direction
    indwhere=np.argmax(np.sin(sliceangle))
    if (np.sin(sliceangle[indwhere])>0.95):
        ISliceMax=np.amax(ImageFilter[np.int(Npixels/2),np.int(Npixels/2):])
        sigmaL=-(radiusL[indwhere]-radii[indwhere])/np.sqrt(2.*np.log(2))
        Gaussian=IsliceMaxYp*np.exp(-(Y-radii[indwhere])*(Y-radii[indwhere])/2./sigmaL/sigmaL)
        condition=((Y>0) & (Y<radii[indwhere]))
        ax2.plot(Y[condition]+d0Y,Gaussian[condition],'g-',lw=1)
        sigmaR=(radiusR[indwhere]-radii[indwhere])/np.sqrt(2.*np.log(2))
        Gaussian=IsliceMaxYp*np.exp(-(Y-radii[indwhere])*(Y-radii[indwhere])/2./sigmaR/sigmaR)
        condition=((Y>0) & (Y>radii[indwhere]))
        ax2.plot(Y[condition]+d0Y,Gaussian[condition],'g-',lw=1)        

    # then the negative Y direction
    indwhere=np.argmin(np.sin(sliceangle))
    if (np.sin(sliceangle[indwhere])<-0.95):
        ISliceMax=np.amax(ImageFilter[np.int(Npixels/2),:np.int(Npixels/2)])
        sigmaL=-(radiusL[indwhere]-radii[indwhere])/np.sqrt(2.*np.log(2))
        Gaussian=IsliceMaxYm*np.exp(-(Y+radii[indwhere])*(Y+radii[indwhere])/2./sigmaL/sigmaL)
        condition=((Y<0) & (np.abs(Y)<radii[indwhere]))
        ax2.plot(Y[condition]+d0Y,Gaussian[condition],'g-',lw=1)
        sigmaR=(radiusR[indwhere]-radii[indwhere])/np.sqrt(2.*np.log(2))
        Gaussian=IsliceMaxYm*np.exp(-(Y+radii[indwhere])*(Y+radii[indwhere])/2./sigmaR/sigmaR)
        condition=((Y<0) & (np.abs(Y)>radii[indwhere]))
        ax2.plot(Y[condition]+d0Y,Gaussian[condition],'g-',lw=1)
        
    ax2.legend(loc='upper right',fontsize='x-small')

    # plot the filtered image on the third panel
    pos3=ax3.contourf(X,Y,np.abs(ImageFilter[:Npixels,:Npixels].T),128,cmap=cm.gist_heat,vmax=1)

    # plot the center
    ax3.plot(d0X,d0Y,'co',ms=10)
    
    # overplot the contour of "radii" measured
    ax3.plot(d0X+radii*np.cos(sliceangle),d0Y+radii*np.sin(sliceangle),'go',ms=2.0)

    # overplot the contour of median "radii" measured
    #ax3.plot(d0X+rad50*np.cos(sliceangle),d0Y+rad50*np.sin(sliceangle),'b-',lw=1)

    # overplot the contour of FWHM measured
    ax3.plot(d0X+radiusL*np.cos(sliceangle),d0Y+radiusL*np.sin(sliceangle),'g--',lw=1)
    ax3.plot(d0X+radiusR*np.cos(sliceangle),d0Y+radiusR*np.sin(sliceangle),'g--',lw=1)

    ax3.set_xlabel(r'$x$ ($M$)')
    ax3.set_ylabel(r'$y$ ($M$)')
    
    # plot the histogram of "readii" on the fourth panel
    ax4.hist(2.*radii,100,histtype='step',density=True,cumulative=True)

    ax4.plot([2.*rad50,2.*rad50],[0,2],'k--',lw=1)
    ax4.plot([2.*rad25,2.*rad25],[0,2],'k:',lw=1)
    ax4.plot([2.*rad75,2.*rad75],[0,2],'k:',lw=1)

    ax4.set_xlabel(r'Image Diameter (M)')
    ax4.set_ylabel(r'Cum. Distribution')
    ax4.set(xlim=[20.,70.])
    ax4.set(ylim=[0,1.1])

    diam=2.*rad50
    # plot the histogram of "FWHM/Diameter" on the fifth panel
    ax5.hist(FWHM/diam,100,histtype='step',density=True,cumulative=True)

    ax5.plot([FWHM50/diam,FWHM50/diam],[0,2],'k--',lw=1)
    ax5.plot([FWHM25/diam,FWHM25/diam],[0,2],'k:',lw=1)
    ax5.plot([FWHM75/diam,FWHM75/diam],[0,2],'k:',lw=1)

    ax5.set_xlabel(r'FWHM/Shadow Diameter')
    ax5.set_ylabel(r'Cum. Distribution')
    ax5.set(xlim=[0.,1])
    ax5.set(ylim=[0,1.1])
    
    # plot the cross sections of the visibility amplitudes on the sixth panel
    
    # normalize visibility by its max
    Visibility/=np.amax(np.abs(Visibility))
    # find the lowest abs u-coordinate (i.e., the location of the vertical cross section)
    umin=np.amin(np.abs(uCo))
    # do the same for the v-coordinate
    vmin=np.amin(np.abs(vCo))
    
    # plot the two cross sections of the Visibility Amplitude
    BlengthHor=uCo[np.abs(vCo)==umin]
    VisHor=np.abs(Visibility.T[np.abs(vCo)==vmin])
    BlengthVer=vCo[np.abs(uCo)==umin]
    VisVer=np.abs(Visibility.T[np.abs(uCo)==umin])
    
    # Find the location of the first minimum in the horizontal direction with
    # baseline length above 1 Glambda (to avoid maximum at zero baseline)
    Bs=BlengthHor[BlengthHor>1]
    zCross=zcr(Bs,np.gradient(VisHor[BlengthHor>1.0]))
    # first minimum
    if (np.size(zCross)>0):
        zFirstHor=zCross[0]
    else:
        zFirstHor=0           # if no mim is found return zero
    
    # Find the location of the first minimum in the vert direction with
    # baseline length above 1 Glambda
    Bs=BlengthVer[BlengthVer>1]
    zCross=zcr(Bs,np.gradient(VisVer[BlengthVer>1.0]))
    # first minimum
    if (np.size(zCross)>0):
        zFirstVer=zCross[0]
    else:
        zFirstVer=0           # if no mim is found return zero        

    ax6.plot(BlengthHor,VisHor,'r-',label='Horizontal')
    ax6.plot(BlengthVer,VisVer,'b-',label='Vertical')
        
    # plot location of first minima
    ax6.plot([zFirstHor,zFirstHor],[1.e-3,10],'r:')
    ax6.plot([zFirstVer,zFirstVer],[1.e-3,10],'b:')
    
    ax6.set_yscale('log')
    ax6.set_xlim([0,10])
    ax6.set_ylim([0.01,2.])
    ax6.set_xlabel(r'Baseline Length (G$\lambda$)')
    ax6.set_ylabel(r'Visibility Amplitude')
    
    ax6.legend(loc='upper right',fontsize='x-small')

    plt.tight_layout()

    plt.savefig(fname=movieframe+'.png')
    plt.close()

    return rad25,rad50,rad75,FWHM25,FWHM50,FWHM75,fracCirc,zFirstHor,zFirstVer

"""!@brief MAIN CODE

@returns nothing

\author UofA Group

\version 2.1

\date October 19, 2021

\bug No known bugs

\warning No known warnings

\todo Nothing left
"""

# set the parameters of the plot
plotParams.setPlotParams()   

# for files in the local directory
path="./"

# start the output file
outfile='datasets_ehtim.out'
asciifile=open(outfile,'w')
    
#get list of .fits files in directory
listoffiles = [x for x in os.listdir(path) if ((x.endswith(".fits")))]
    
index=1
    
asciifile.write("# filename diameter50 diameter25 diameter75 FWHM50 FWHM25 FWHM75 fraccirc zFirstHor zFirstVer\n")
    
# read FITS file
for file_name in listoffiles:

    hdulist = fits.open(path+file_name)
    hdu     = hdulist[0]
    array   = hdu.data
    pixel   = np.abs(hdu.header['CDELT1'])*3.6e9      # convert degrees to uas
    hdulist.close()

    # uncomment this for SMILI outputs, which include pol maps
    #    Npts=np.shape(array)[2]                           # size of each dimension
    Npts=np.shape(array)[0]                           # size of each dimension
    FOV=(np.float(Npts-1))*pixel                      # field of view
        
    #x-y grids
    X=np.linspace(-FOV/2,FOV/2,num=Npts,endpoint=True)
    Y=np.linspace(-FOV/2,FOV/2,num=Npts,endpoint=True)
    
    # image

    # uncomment this for SMILI
    #I = array[0,0,:,:]
    I = array
    I = I.T
        
    # find the center by minimizing the rms of the diameters
    d0X,d0Y=findcenter(Npts,X,Y,I)
        
    # label for the image (replacing the underscores with spaces)
    label1=file_name.replace('_',' ')
    label=label1.replace('.fits','')
    # frame number for movie
    movieframe=str(index).zfill(4)

    # characterize this image and get its parameters
    rad25,rad50,rad75,FWHM25,FWHM50,FWHM75,fracCirc,zFirstHor,zFirstVer=char_image(label,movieframe,Npts,X,Y,I)
        
    """
    # for more verbose output
    print("file: ",file_name)
    print("Center at X=",d0X,' Y=',d0Y)
    print("diameter=",2.*rad50,"-",2.*(rad50-rad25),"+",2.*(rad75-rad50))
    print("FWHM=",FWHM50,"-",2.*(FWHM50-FWHM25),"+",2.*(FWHM75-FWHM50))
        
    """
        
    # output results in file
    listResults=[file_name, 2.*rad50, 2.*rad25, 2.*rad75, FWHM50, FWHM25, FWHM75,fracCirc,zFirstHor,zFirstVer]
    res = " ".join([str(i) for i in listResults]) 
    asciifile.write(str(res)+"\n")
    
    index+=1


# close output file
asciifile.close()
    
