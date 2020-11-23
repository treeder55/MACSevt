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
from numba import int64, int32, float32, boolean, jitclass, char, njit, jit
import numba
from mpl_toolkits.mplot3d import Axes3D
from evtParsefunctions import *
import TRplot as trp

def bin2angle(x):
    return x*184/1000
def angle2bin(x):
    return x*1000/184
def time2angle(x):
    return x*184/62.926728468
def angle2time(x):
    return x*62.926728468/184

def plottrace(tmt0coverage,A3coverage):
    fig = plt.figure()
    ax2 = fig.add_axes([0,0,1.6,1.4])
    ax2.plot(A3coverage,tmt0coverage,'.')
    ax2.set_xlabel('A3', fontsize = 20)
    ax2.set_ylabel('t-t0', fontsize = 20)

class MACS_BS:
    def __init__(self, varss):
     #   rottime = 62.926728468*2                            # replace this with first pass through data to get the rotation period and t0 period.
        self.filename=varss[0]; self.lowang=varss[1]; self.highang=varss[2]; self.BSbins=varss[3]; self.offset=varss[4]; self.fpx=varss[5]; self.BSlpulse=varss[6]; 
        self.BShpulse=varss[7]; self.rottime=varss[8]; self.tstepforcoverage=varss[9]; self.omega=varss[10]
        self.files,valuescell = unpackevts(self.filename)
        self.E,self.raw_t,self.BSpers,self.t_BSl,self.t_BSh,self.inds = {},{},{},{},{},{}
        for nn in valuescell.keys():
            print(self.files[nn])
            print('values/4 = ' + str(len(valuescell[nn])/4))
            self.raw_t[nn],ID,self.inds[nn],self.simcov                              = Unpackvaluescell(valuescell[nn],self.tstepforcoverage)
            self.t_BSl[nn],self.t_BSh[nn]                                 = MakeBS(self.inds[nn],ID,self.BSlpulse,self.BShpulse,self.raw_t[nn],self.rottime)
            difft_BSl = self.t_BSl[nn][1:]-self.t_BSl[nn][:-1]; difft_BSh = self.t_BSh[nn][1:]-self.t_BSh[nn][:-1];
            print('missing l pulse: '+str(np.where(difft_BSl>self.rottime*1.5)[0]))
            print('missing h pulse: '+str(np.where(difft_BSh>self.rottime*1.5)[0]))
            print('double l pulse: '+str(np.where(difft_BSl<self.rottime*0.5)[0]))
            print('double h pulse: '+str(np.where(difft_BSh<self.rottime*0.5)[0]))
    def histogram(self):
        for nn in self.raw_t.keys():
            difft_BSl,self.t_BSl[nn],self.inds[nn][self.BSlpulse] = pulsecorrect(self.t_BSl[nn],self.inds[nn][self.BSlpulse],self.rottime); 
            difft_BSh,self.t_BSh[nn],self.inds[nn][self.BShpulse] = pulsecorrect(self.t_BSh[nn],self.inds[nn][self.BShpulse],self.rottime);
            self.E[nn],self.BSpers[nn],self.BSbinedges =  MakeEhistacc1d(self.raw_t[nn],self.BSlpulse,self.BShpulse,self.BSbins,self.lowang,self.highang,self.t_BSl[nn],self.inds[nn],self.omega,self.simcov)
            print('time of file = ' + str(self.raw_t[nn][-1]))
            #self.E[nn]                                           = MakeCov1d(self.E[nn],self.raw_t[nn],self.tstepforcoverage,self.t_BSl[nn],self.BSbinedges,self.lowang,self.highang,self.omega)
            trp.trplot(np.arange(len(self.E[nn][23])),self.E[nn][23],labelcolor='dodgerblue',ylabel='weight',xlabel='bin')
            #self.inds[nn] = inds; self.E[nn] = E; self.t_t0[nn] = t_t0; self.t_BS[nn] = t_BS; self.raw_t[nn] = raw_t; self.tpers[nn] = tpers; self.BSpers[nn] = BSpers
    def savehistasrg0(self,*args):
        rg0directory = args[0]
        for nn in self.E.keys():
            dupE1d = np.zeros([len(self.E[nn]),2,len(self.E[nn][0])])
            for i in range(len(self.E[nn])):
                for j in range(len(dupE1d[i])):
                    dupE1d[i][j] = np.divide(self.nE[nn][i],2) # to conserve total counts in file
            fname = rg0directory + self.fpx + '_' + str(nn) + '.rg0'
            flatE = np.concatenate([self.E[nn][:20].flatten(),self.E[nn][23].flatten()])
            header = np.array([1,0,2,1,self.lowang,self.highang,self.BSbins,self.offset])
            Earray = np.concatenate([header,flatE])
            binaryE = struct.pack('f'*len(Earray),*Earray)
            file = open(fname,'wb')
            file.write(binaryE)
            file.close
    def saveduphistasrg0(self,*args):
        rg0directory = args[0]
        self.pulsewidth = args[1]
        self.tbins = args[2]
        self.dE = {}
        for nn in self.E.keys():
            self.dE[nn] = np.zeros([25,self.tbins,self.BSbins])
            for i in range(len(self.E[nn])):
                for j in range(len(self.dE[nn][i])):
                    self.dE[nn][i][j] = np.divide(self.E[nn][i],self.tbins)
            fname = rg0directory + self.fpx + '_' + str(nn) + '.rg0'
            flatE = np.concatenate([self.dE[nn][:20].flatten(),self.dE[nn][23].flatten()])
            header = np.array([1,0,self.pulsewidth,self.tbins,self.lowang*1.0,self.highang*1.0,self.BSbins,self.offset])
            Earray = np.concatenate([header,flatE])
            binaryE = struct.pack('f'*len(Earray),*Earray)
            file = open(fname,'wb')
            file.write(binaryE)
            file.close
    def duplicate(self,*args):
        self.tbins = args[0]
        self.pulsewidth = args[1]
        dupE1d = {}
        for nn in E1d.keys():
            dupE1d[nn] = np.zeros([25,self.tbins,self.A3bins])
            for i in range(21):
                for j in range(len(dupE1d[nn][i])):
                    dupE1d[nn][i][j] = np.divide(self.nE[nn][i],self.tbins)
        self.dupnE = dupE1d
    def duplicate_imposecoverage(self,*args):
        self.tbins = args[0]
        self.pulsewidth = args[1]
        dupE1d = {}
        for nn in E1d.keys():
            dupE1d[nn] = np.zeros([25,self.tbins,self.A3bins])
            for i in range(21):
                for j in range(len(dupE1d[nn][i])):
                    dupE1d[nn][i][j] = np.divide(self.nE[nn][i],self.tbins)
            dupE1d[nn] =dupE1d[nn]*self.E[nn][23]
            dupE1d[nn] =dupE1d[nn]/self.E[nn][23]
            rationonzerotototal = len(self.E[nn][23][np.where(self.E[nn][23]>0)])/len(self.E[nn][23].flatten())
            dupE1d[nn] = dupE1d[nn]/rationonzerotototal  
        self.dupnE = dupE1d
    def plotE(self,index = [0],xlim = 'auto',ylim='auto',off = 0,labelcolor = 'dodgerblue',fs = [20,30],figsize=[2,2]):
        self.time={}
        for nn in self.E.keys():
            self.time[nn] = np.arange(1,self.BSbins+1)*self.rottime/(2*self.BSbins)
            for j in index:
                fig, ax = plt.subplots(constrained_layout=True,figsize=(figsize[0],figsize[1]))
                ax.plot(self.time[nn],self.E[nn][j],'-o',color='black',markersize = 2,linewidth=0.5,markeredgewidth = 1)
                ax.set_xlabel('time (s)',color=labelcolor,fontsize = fs[1])
                ax.set_ylabel('counts',color=labelcolor,fontsize = fs[1])
                ax.set_title('spec ch = '+str(j+1)+', index='+str(j)+', file='+str(nn),color=labelcolor,fontsize=fs[0])
                ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=fs[0])
                ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=fs[0])
                ax.grid(True)
                ax.minorticks_on()
                ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
                ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
                secax = ax.secondary_xaxis('top',functions = (time2angle,angle2time))
                secax.set_xlabel('angle (degree)',color=labelcolor,fontsize=fs[1])
                secax.tick_params(axis='x',labelcolor=labelcolor,color=labelcolor,labelsize=fs[0])
                if ylim != 'auto':
                    ax.set_ylim([ylim[0],ylim[1]])
                if xlim != 'auto':
                    ax.set_xlim([xlim[0],xlim[1]])
                plt.show()

class MACS_BS_t:
    def __init__(self, varss):
        self.filename=varss[0]; self.pulsewidth=varss[1]; self.lowang=varss[2]; self.highang=varss[3]; self.tbins=varss[4]; self.BSbins=varss[5]; self.offset=varss[6]; 
        self.fpx=varss[7]; tpulse=varss[8]; BSlpulse=varss[9]; BShpulse=varss[10]; rottime=varss[11]; tstepforcoverage=varss[12]; omega=varss[13];
        self.files,valuescell = unpackevts(self.filename)
        self.E,self.raw_t,self.tpers,self.BSpers,self.t_t0,self.t_BSl,self.t_BSh,self.inds,self.ID = {},{},{},{},{},{},{},{},{}
        for nn in valuescell.keys():
            print(self.files[nn])
            print('values/4 = ' + str(len(valuescell[nn])/4))
            raw_t,ID,inds,simcov  = Unpackvaluescell(valuescell[nn],tstepforcoverage)
   #         self.inds[nn] = inds; self.raw_t[nn] = raw_t; self.ID[nn] = ID; 
   # def asdf():
            print('time of file = ' + str(raw_t[-1]))
            t_BSl,t_BSh = MakeBS(inds,ID,BSlpulse,BShpulse,raw_t,rottime); t_t0 = Maket_t0(raw_t,inds,tpulse);
            difft_BSl = t_BSl[1:]-t_BSl[:-1]; difft_BSh = t_BSh[1:]-t_BSh[:-1]; difft_t0 = t_t0[1:]-t_t0[:-1];
            print('missing l pulse: '+str(np.where(difft_BSl>rottime*1.5)[0]))
            print('missing h pulse: '+str(np.where(difft_BSh>rottime*1.5)[0]))
            print('missing t0 pulse: '+str(np.where(difft_t0>self.pulsewidth*1.5)[0]))
            print('double l pulse: '+str(np.where(difft_BSl<rottime*0.5)[0]))
            print('double h pulse: '+str(np.where(difft_BSh<rottime*0.5)[0]))
            print('double t0 pulse: '+str(np.where(difft_t0<self.pulsewidth*0.5)[0]))
            difft_t0,t_t0,inds[tpulse] = pulsecorrect(t_t0,inds[tpulse],self.pulsewidth); 
            difft_BSl,t_BSl,inds[BSlpulse] = pulsecorrect(t_BSl,inds[BSlpulse],rottime); difft_BSh,t_BSh,inds[BShpulse] = pulsecorrect(t_BSh,inds[BShpulse],rottime);
#        self.raw_t=raw_t;self.tpulse=tpulse;self.BSlpulse=BSlpulse;self.BShpulse=BShpulse;self.t_t0=t_t0;self.t_BSl=t_BSl;self.inds=inds;self.omega=omega
            E,tpers,BSpers,tbinedges,BSbinedges     = MakeEhistacc2d(raw_t,tpulse,BSlpulse,BShpulse,self.tbins,self.BSbins,self.lowang,self.highang,t_t0,t_BSl,inds,omega,simcov)
            print('sum of detector events = ' + str(np.sum(E[:20])))
            trp.plothist(h=E[23],title='coverage histogram',aspect=self.BSbins/self.tbins)
            self.E[nn] = E; self.t_BSl[nn] = t_BSl; self.t_BSh[nn] = t_BSh; self.inds[nn] = inds; self.t_t0[nn] = t_t0; self.raw_t[nn] = raw_t; self.tpers[nn] = tpers; self.BSpers[nn] = BSpers
    def integratehistintime(self):
        self.tE = {}
        for nn in self.E.keys():
            tE = np.zeros((np.shape(self.E[nn])[0],np.shape(self.E[nn])[2]))
            A = {}
            for ch in range(len(self.E[nn])):
                A[ch] = np.transpose(self.E[nn][ch])
                tE[ch] = np.zeros(len(A[ch]))
                for AA3 in range(len(A[ch])):
                    tE[ch][AA3] = np.sum(A[ch][AA3])
                    tE[ch] = np.transpose(tE[ch])
            self.tE[nn] = tE
    def integratehistinA3(self):
        A3E = np.zeros((np.shape(self.nE)[0],np.shape(self.nE)[1]))
        for ch in range(len(self.nE)):
            A3E[ch] = np.zeros(len(self.nE[ch]))
            for AA3 in range(len(self.nE[ch])):
                A3E[ch][AA3] = np.sum(self.nE[ch][AA3])
                #tE[ch] = np.transpose(tE[ch])
        self.A3E = A3E
    def savehistasrg0(self,*args):
        rg0directory = args[0]
        for nn in self.E.keys():
            fname = rg0directory + self.fpx + '_' + str(nn) + '.rg0'
            flatE = np.concatenate([self.E[nn][:20].flatten(),self.E[nn][23].flatten()])
            header = np.array([1,0,self.pulsewidth,self.tbins,self.lowang*1.0,self.highang*1.0,self.BSbins,self.offset])
            Earray = np.concatenate([header,flatE])
            binaryE = struct.pack('f'*len(Earray),*Earray)
            file = open(fname,'wb')
            file.write(binaryE)
            file.close
    def savepartialtimeasrg0(self,rg0directory,t1,t2):
        self.dE = {}
        for nn in self.E.keys():
            self.dE[nn] = np.zeros(np.shape(self.E[nn]))
            tempbgE = np.zeros([len(self.E[nn]),self.BSbins,self.tbins])
            for i in range(len(self.E[nn])): # loop through the 25 histograms, one for each time stamping channel.
                temp = np.transpose(self.E[nn][i]) # transpose the histograms
                for k in range(len(temp)): # loop through each A3 column of a histogram corresponding to a 1d array along time axis
                    temp2 = temp[k][int(self.tbins/self.pulsewidth*t1):int(self.tbins/self.pulsewidth*t2)]  # take last 8 seconds, (last 80 bins)
                    tempbgE[i][k] = tempbgE[i][k]+np.average(temp2)
                self.dE[nn][i] = np.transpose(tempbgE[i])
            fname = rg0directory + self.fpx + '_' + str(nn) + '.rg0'
            flatE = np.concatenate([self.dE[nn][:20].flatten(),self.dE[nn][23].flatten()])
            header = np.array([1,0,self.pulsewidth,self.tbins,self.lowang*1.0,self.highang*1.0,self.BSbins,self.offset])
            Earray = np.concatenate([header,flatE])
            binaryE = struct.pack('f'*len(Earray),*Earray)
            file = open(fname,'wb')
            file.write(binaryE)
            file.close
    def saveduphistasrg0(self,*args):
        rg0directory = args[0]
        for nn in self.nE.keys():
            fname = rg0directory + self.fpx + '_' + str(nn) + '.rg0'
            #flatE = np.concatenate([nE[:-5].flatten(),nE[-1].flatten()])
            flatE = self.dupnE[nn][:-5].flatten()
            flatE[np.isnan(flatE)] = -999
            flatE[np.isinf(flatE)] = -999
            header = np.array([1,0,self.pulsewidth,np.round((self.pulsewidth)/self.tbins,2),self.lowang,self.highang,np.round((self.highang-self.lowang)/self.A3bins,2),self.offset])
            #header = np.array([1,lowtedge,hightedge,tbin,lowang,highang,A3bin,offset])
            Earray = np.concatenate([header,flatE])
            binaryE = struct.pack('f'*len(Earray),*Earray)
            file = open(fname,'wb')
            file.write(binaryE)
            file.close
    def savehistasrg0_old(self,*args):
        rg0directory = args[0]
        for nn in self.nE.keys():
            self.tpers[nn] = np.concatenate([self.tpers[nn][:12],self.tpers[nn][14:-5]])
            self.A3pers[nn] = np.concatenate([self.A3pers[nn][:12],self.A3pers[nn][14:-5]])
            self.tpers[nn] = np.transpose(self.tpers[nn])
            self.A3pers[nn] = np.transpose(self.A3pers[nn])
            fname = rg0directory + self.fpx + '_' + str(nn) + '.rg0'
            #flatE = np.concatenate([nE[:-5].flatten(),nE[-1].flatten()])
            flatE = self.nE[nn][:-5].flatten()
            flatE[np.isnan(flatE)] = -999
            flatE[np.isinf(flatE)] = -999
            header = np.array([1,0,self.pulsewidth,(self.pulsewidth*1.0)/self.tbins,self.lowang*1.0,self.highang*1.0,(self.highang-self.lowang*1.0)/self.A3bins,self.offset])
            #header = np.array([1,np.average(self.tpers[nn][0]),np.average(self.tpers[nn][1]),np.average(self.tpers[nn][2]),np.average(self.A3pers[nn][0]),np.average(self.A3pers[nn][1]),np.average(self.A3pers[nn][2]),self.offset])

            #header = np.array([1,lowtedge,hightedge,tbin,lowang,highang,A3bin,offset])
            Earray = np.concatenate([header,flatE])
            binaryE = struct.pack('f'*len(Earray),*Earray)
            file = open(fname,'wb')
            file.write(binaryE)
            file.close
    def duplicate(self,*args):
        self.tbins = args[0]
        self.pulsewidth = args[1]
        self.dE = {}
        for nn in E1d.keys():
            self.dE[nn] = np.zeros([25,self.tbins,self.A3bins])
            for i in range(21):
                for j in range(len(self.dE[nn][i])):
                    self.dE[nn][i][j] = np.divide(self.nE[nn][i],self.tbins)
    def plotE(self,index = [0],xlim = 'auto',ylim='auto',off = 0,labelcolor = 'dodgerblue',fs = [20,30],figsize=[2,2]):
        self.time={}
        for nn in self.tE.keys():
            self.time[nn] = np.arange(1,self.BSbins+1)*self.rottime/(2*self.BSbins)
            for j in index:
                fig, ax = plt.subplots(constrained_layout=True,figsize=(figsize[0],figsize[1]))
                ax.plot(self.time[nn],self.tE[nn][j],'-o',color='black',markersize = 2,linewidth=0.5,markeredgewidth = 1)
                ax.set_xlabel('time (s)',color=labelcolor,fontsize = fs[1])
                ax.set_ylabel('counts',color=labelcolor,fontsize = fs[1])
                ax.set_title('spec ch = '+str(j+1)+', index='+str(j)+', file='+str(nn),color=labelcolor,fontsize=fs[0])
                ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=fs[0])
                ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=fs[0])
                ax.grid(True)
                ax.minorticks_on()
                ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
                ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
                secax = ax.secondary_xaxis('top',functions = (time2angle,angle2time))
                secax.set_xlabel('angle (degree)',color=labelcolor,fontsize=fs[1])
                secax.tick_params(axis='x',labelcolor=labelcolor,color=labelcolor,labelsize=fs[0])
                if ylim != 'auto':
                    ax.set_ylim([ylim[0],ylim[1]])
                if xlim != 'auto':
                    ax.set_xlim([xlim[0],xlim[1]])
                plt.show()
    def plotcovhist3d(self,theta,phi,index):
        delta_A3=self.highang-self.lowang
        X,Y = np.meshgrid(np.arange(0,delta_A3,delta_A3/self.A3bins),np.arange(0,self.pulsewidth,self.pulsewidth/self.tbins))
        for nn in self.nE.keys():
            plt.figure(figsize = (30,12))
            ax = plt.axes(projection = '3d')
            ax.plot_surface(Y,X,self.E[nn][index],cmap='ocean', rstride=1, cstride=1, linewidth=0,vmax=np.max(self.E[nn][index]))#vmin = np.min(hist),vmax = np.max(hist),aspect='equal',cmap='rainbow')    
            ax.set_zlim(0,np.max(self.E[nn][index]))
            ax.view_init(theta, phi)
            ax.set_ylabel('BS',fontsize = 20)
            ax.set_xlabel('t',fontsize = 20)
            ax.set_zlabel('norm weight',fontsize = 20)

class readrg0():
    def __init__(self,filename):
        if filename[-4:] == '.rg0':
            file = open(filename, mode = 'rb')
            self.filecontent = file.read()
            self.header = struct.unpack('8f',self.filecontent[:32])
            self.tbins = int(np.round(self.header[2]/self.header[3])); self.BSbins = int(np.round((self.header[5]-self.header[4])/self.header[6])); self.lowang = self.header[4]; self.highang = self.header[5];
            self.offset = self.header[7]
            self.values = struct.unpack('f'*21*self.tbins*self.BSbins,self.filecontent[32:])
            EE = np.array(self.values)
            self.E = EE.reshape((21,300,1000))
    def countzeros(self):
        c = self.E[20].flatten()
        d = self.E[:20].flatten()
        f = np.where(c==0)[0]
        g = np.where(self.E[20]==0)
        print('zeros in coverage = ' +str(len(np.where(c==0)[0])))
        print('zeros in det hists = '+str(len(np.where(d==0)[0])))
        for i in range(len(self.E)-1):
            b = self.E[i].flatten()
            print('channel = '+str(i))
            print(np.max(b[c==0]))
            print(np.max(b[f]))
            print(np.max(self.E[i][g]))

# Things to do:
# -> Turn into class
# -> simplify normalization routines
# -> keep nans as nan when writing binary file
# -> make function to check the total counts of events before operating and after normalization
