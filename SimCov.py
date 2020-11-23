import numpy as np
import scipy as sp
import struct
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator)
import time
import sys

def make1Dcoverage(delta_BS,delta_tBS,totaltimeofevt,BS_lag,BSbins,BSstart): # delta_t is the pulse period, delta_A3 is the A3_max-A3_min angles in degrees, 
    # delta_tA3 is the total time that passes during a full delta_A3 rotation, totaltimeofevt is the total time of the event file, A3_lag is the lag time between stopping and starting a rotation, tbins is the number of time bins for the histogram, A3bins the number of A3 bins, A3start and tstart are the initial times of the scan. I think the coverage doesn't depend on the initial values so the starting parameters can probably be removed from the scripts.
    tpoints = np.arange(0,totaltimeofevt,delta_tBS/BSbins/40)                 # array of times with small seperation, 40 times per bin
################################################################################
    BS = np.arange(BSstart,BSstart+totaltimeofevt+2*delta_tBS,delta_tBS+BS_lag)   # array of times at which you would receive BS pulses
    sizeofhalfBS = np.int(np.ceil(len(BS)/2))
    if sizeofhalfBS == len(BS)/2:                                          # duplicate each time value, making an even number of times, so terminate one if need be.
        BS1 = np.insert(np.zeros(sizeofhalfBS),np.arange(0,sizeofhalfBS,1)+1,np.zeros(sizeofhalfBS)+delta_BS)
    else:
        BS1 = np.insert(np.zeros(sizeofhalfBS),np.arange(0,sizeofhalfBS,1)+1,np.zeros(sizeofhalfBS)+delta_BS)
        BS1 = BS1[:-1]
    BS = np.concatenate([[BS[0]-delta_tBS],BS])
    BS1 = np.concatenate([[BS1[1]],BS1])
    BScoverage = np.interp(tpoints,BS,BS1)
    pixeltime = delta_tBS/BSbins
################################################################################
    hist = np.histogram(BScoverage,bins = BSbins)[0]*1/40*pixeltime       # normalized to represent amount of time spent in each pixel
################################################################################  
    return BScoverage,hist
def make2Dcoverage(delta_t,delta_BS,delta_tBS,totaltimeofevt,BS_lag,tbins,BSbins,BSstart,tstart): # delta_t is the pulse period, delta_A3 is the A3_max-A3_min angles in degrees, 
    # delta_tA3 is the total time that passes during a full delta_A3 rotation, totaltimeofevt is the total time of the event file, A3_lag is the lag time between stopping and starting a rotation, tbins is the number of time bins for the histogram, A3bins the number of A3 bins, A3start and tstart are the initial times of the scan. I think the coverage doesn't depend on the initial values so the starting parameters can probably be removed from the scripts.
    t0 = np.arange(tstart,tstart+totaltimeofevt+2*delta_t,delta_t)         # array of times at which you receive a t0 pulse during the scan
    tt0 = np.insert(t0,np.arange(1,len(t0)+1,1),t0)                        # making each time happen twice in the array
    tt0 = tt0[1:-1]                                                        # get rid of first and last values    
    difft = np.zeros(len(t0))+delta_t                                      # array of differences between each t0 time
    tt = np.insert(difft,np.arange(0,len(difft),1),np.zeros(len(difft)))   # putting a zero in front of each difference value
    tt = tt[:-2]
    #tt = np.concatenate([tt,[0,totaltimeofevt]]) This one corrects for a missing pulse at the end of the tt, but I don't think there is one missing in this function.
    tpoints = np.arange(0,totaltimeofevt,delta_t/tbins/40)                 # array of times with small seperation, 40 times per time bin
    tmt0coverage = np.interp(tpoints,tt0,tt)                               # the array to be used for histograming
################################################################################
    BS = np.arange(BSstart,BSstart+totaltimeofevt+2*delta_tBS,delta_tBS+BS_lag)   # array of times at which you would receive BS pulses
    sizeofhalfBS = np.int(np.ceil(len(BS)/2))
    if sizeofhalfBS == len(BS)/2:                                          # duplicate each time value, making an even number of times, so terminate one if need be.
        BS1 = np.insert(np.zeros(sizeofhalfBS),np.arange(0,sizeofhalfBS,1)+1,np.zeros(sizeofhalfBS)+delta_BS)
    else:
        BS1 = np.insert(np.zeros(sizeofhalfBS),np.arange(0,sizeofhalfBS,1)+1,np.zeros(sizeofhalfBS)+delta_BS)
        BS1 = BS1[:-1]
    BS = np.concatenate([[BS[0]-delta_tBS],BS])
    BS1 = np.concatenate([[BS1[1]],BS1])
    BScoverage = np.interp(tpoints,BS,BS1)
################################################################################
    pixeltime = delta_t/tbins
    hist = np.histogram2d(tmt0coverage,BScoverage,bins = [tbins,BSbins])[0]*1/40*pixeltime # normalized to represent amount of time spent in each pixel
################################################################################  
    return tmt0coverage,BScoverage,hist
def plothist3d(hist,delta_A3,A3bins,delta_t,tbins,theta,phi):
    plt.figure(figsize = (30,12))
    ax = plt.axes(projection = '3d')
    X,Y = np.meshgrid(np.arange(0,delta_A3,delta_A3/A3bins),np.arange(0,delta_t,delta_t/tbins))
    ax.plot_surface(Y,X,hist,cmap='ocean', rstride=1, cstride=1, linewidth=0,vmax=np.max(hist))#vmin = np.min(hist),vmax = np.max(hist),aspect='equal',cmap='rainbow')    
    ax.set_zlim(0,np.max(hist))
    ax.view_init(theta, phi)
    ax.set_ylabel('BS',fontsize = 20)
    ax.set_xlabel('t',fontsize = 20)
    ax.set_zlabel('time spent in pixel',fontsize = 20)
    #cbar = fig.colorbar(im,shrink = 0.8)
def plothist2d(hist):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1.6,1.4])
    im = ax.imshow(hist,vmax=np.max(hist),cmap='rainbow',aspect=1)
    ax.set_xlabel('BS',fontsize = 20)
    ax.set_ylabel('t',fontsize = 20)
    cbar = fig.colorbar(im,shrink = 0.8)
    plt.show()
def plothist1d(hist1d,delta_BS):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1.6,1.4])
    ax.plot(np.arange(0,delta_BS,delta_BS/len(hist1d)),hist1d,'k.')
    ax.set_xlabel('BS',fontsize = 20)
    ax.set_ylabel('t',fontsize = 20)
    plt.show()

