import numpy as np
import scipy as sp
import struct
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import BSacc2 as bsa


def Maket_A3(t_A31, t_A32,lowang,highang):
    bb = np.concatenate([np.zeros(len(t_A31))+lowang,np.zeros(len(t_A32))+highang])
    aa = np.concatenate([t_A31,t_A32])
    bb = bb[aa.argsort()]
    aa = aa[aa.argsort()]
    bb = np.concatenate([[bb[1]],bb,[bb[-2]]])
    aa = np.concatenate([[aa[0] - np.average(aa[1:]-aa[:-1])],aa,[aa[-1] + np.average(aa[1:]-aa[:-1])]])
    t_A3 = np.array([aa,bb])
    return t_A3
def addextraendpulses(tp):
    diff = np.average(tp[1:] - tp[:-1])
    tp = np.concatenate([[tp[0] - diff],tp,[tp[-1] + diff]])
    return tp
def Maket_t0(raw_t,inds,tpulse):
    indexoft0 = inds[tpulse]
    t_t0 = raw_t[indexoft0]
    return t_t0
def MakeBS(inds,ID,BSlpulse,BShpulse,raw_t,rottime):
    if len(np.where(ID==BSlpulse)[0])==0:
        t_A32 = raw_t[inds[BShpulse]]
        if len(t_A32) ==1:
            t_A32 = np.array([t_A32[0],rottime+t_A32[0]])
        t_A31 = t_A32+rottime/2
        t_A31 = np.concatenate([[t_A31[0]-rottime],t_A31])
        t_A31 = np.concatenate([t_A31,[t_A31[-1]+rottime]])
        t_A31 = np.delete(t_A31,np.where(t_A31<0))
        t_A32 = np.delete(t_A32,np.where(t_A32<0))
        t_A31 = np.delete(t_A31,np.where(t_A31>raw_t[-1]))
        t_A32 = np.delete(t_A32,np.where(t_A32>raw_t[-1]))
    elif len(np.where(ID==BShpulse)[0])==0:
        t_A31 = raw_t[inds[BSlpulse]]
        if len(t_A31) ==1:
            t_A31 = np.array([t_A31[0],rottime+t_A31[0]])
        t_A32 = t_A31+rottime/2
        t_A32 = np.concatenate([[t_A32[0]-rottime],t_A32])
        t_A32 = np.concatenate([t_A32,[t_A32[-1]+rottime]])
        t_A31 = np.delete(t_A31,np.where(t_A31<0))
        t_A32 = np.delete(t_A32,np.where(t_A32<0))
        t_A31 = np.delete(t_A31,np.where(t_A31>raw_t[-1]))
        t_A32 = np.delete(t_A32,np.where(t_A32>raw_t[-1]))
    else:
        t_A32 = raw_t[inds[BShpulse]]
        t_A31 = raw_t[inds[BSlpulse]]
    return t_A31,t_A32
def MakeEhistacc1d(raw_t,BSlpulse,BShpulse,BSbins,lowang,highang,t_BSl,inds,omega,sumcov):
    add = {}; add[0] = 24; add[BSlpulse] = 20; add[BShpulse] = 21; add[50] = 23;
    for i in np.arange(1,41,2): add[i] = int((i-1)/2);
    E = np.zeros((25,BSbins)); BSpers = np.zeros((25,3));
    trot = np.average(t_BSl[1:]-t_BSl[:-1]);
    BSbinedges = np.linspace(lowang,highang,BSbins+1);
    BS = bsa.fbs(raw_t,t_BSl,inds[BSlpulse],omega,lowang,highang,trot);
    for i in inds.keys():
        Ehists = np.histogram(BS[inds[i]],bins = BSbinedges)
        E[add[i]] = Ehists[0].astype(np.float32)
        BSpers[add[i]] = np.array([Ehists[1][0],Ehists[1][-1],(Ehists[1][-1]-Ehists[1][0])/BSbins])
    # sumcov = np.sum(E[23])
    print('sum of coverage = '+str(sumcov))
    E[23] = np.divide(E[23],sumcov)
    print('sum of coverage after reducing = '+str(np.sum(E[23])))
    return E,BSpers,BSbinedges
def MakeEhistacc2d(raw_t,tpulse,BSlpulse,BShpulse,tbins,BSbins,lowang,highang,t_t0,t_BSl,inds,omega,sumcov):
    add = {}; add[0] = 24; add[BSlpulse] = 20; add[BShpulse] = 21; add[tpulse] = 22; add[50] = 23;
    for i in np.arange(1,41,2): add[i] = int((i-1)/2);
    E = np.zeros((25,tbins,BSbins)); tpers = np.zeros((25,3)); BSpers = np.zeros((25,3));
    period = np.average(t_t0[1:]-t_t0[:-1]); trot = np.average(t_BSl[1:]-t_BSl[:-1]);
    tbinedges = np.linspace(0,period,tbins+1); BSbinedges = np.linspace(lowang,highang,BSbins+1);
    t = bsa.ft(raw_t,t_t0,inds[tpulse],period); BS = bsa.fbs(raw_t,t_BSl,inds[BSlpulse],omega,lowang,highang,trot);
    for i in inds.keys():
        Ehists = np.histogram2d(t[inds[i]],BS[inds[i]],bins = [tbinedges,BSbinedges])
        E[add[i]] = Ehists[0].astype(np.float32)
        tpers[add[i]] = np.array([Ehists[1][0],Ehists[1][-1],(Ehists[1][-1]-Ehists[1][0])/tbins])
        BSpers[add[i]] = np.array([Ehists[2][0],Ehists[2][-1],(Ehists[2][-1]-Ehists[2][0])/BSbins])
    ##sumcov = len(tcov) #np.sum(E[23])
    print('sum of coverage = '+str(sumcov))
    E[23] = np.divide(E[23],sumcov)
    print('sum of coverage after reducing = '+str(np.sum(E[23])))
    return E,tpers,BSpers,tbinedges,BSbinedges
def MakeEhist04202020(raw_t,tpulse,BSlpulse,BShpulse,tbins,A3bins,pulsewidth,lowang,highang,tt_t0,t_A3,inds):
    add = {}
    add[0] = 24
    for i in np.arange(1,41,2):
        add[i] = int((i-1)/2)
    add[BSlpulse] = 20
    add[BShpulse] = 21
    add[tpulse] = 22
    E = np.zeros((25,tbins,A3bins))
    tpers = np.zeros((25,3))
    A3pers = np.zeros((25,3))
#    pulsewidth = np.average(difft0)
    #tbinedges = np.linspace(0,pulsewidth+pulsewidth/tbins,tbins+1)
    #A3binedges = np.linspace(lowang-(highang-lowang)/A3bins,highang+(highang-lowang)/A3bins,A3bins+1)
    tbinedges = np.linspace(0,pulsewidth,tbins+1)
    A3binedges = np.linspace(lowang,highang,A3bins+1)
    tmt0 = np.interp(raw_t,tt_t0[0],tt_t0[1])
    A3array = np.interp(raw_t,t_A3[0],t_A3[1])
    for i in inds.keys():
        #Earray[add[i]] = np.array([ntmt0[ninds[i]],A3array[ninds[i]]])
        #Ehists = np.histogram2d(tmt0[inds[i]],A3array[inds[i]],bins = [tbinedges,A3binedges])
        Ehists = np.histogram2d(tmt0[inds[i]],A3array[inds[i]],bins = [tbinedges,A3binedges])
        E[add[i]] = Ehists[0]*1.0
        tpers[add[i]] = np.array([Ehists[1][0],Ehists[1][-1],(Ehists[1][-1]-Ehists[1][0])/tbins])
        A3pers[add[i]] = np.array([Ehists[2][0],Ehists[2][-1],(Ehists[2][-1]-Ehists[2][0])/A3bins])
    return E,tpers,A3pers,tbinedges,A3binedges,add,tmt0
def pulsecorrect(t_t0,t0inds,period):
    difft_t0 = t_t0[1:]-t_t0[:-1]
    B = np.concatenate([[True],(difft_t0>period*0.8)&(difft_t0<period*1.2)]) # The extra true is needed to match the indexes up
    t0inds = t0inds[B]
    t_t0 = t_t0[B]
    difft_t0 = difft_t0[B[1:]]
    t_t0 = np.concatenate([[t_t0[0]-np.average(difft_t0)],t_t0,[t_t0[-1]+np.average(difft_t0)]])#,[t_t0[-1]+avgdiff]])
    difft_t0 = t_t0[1:]-t_t0[:-1]
    return difft_t0,t_t0,t0inds
#def t0correct(t_t0,t0inds,period):
#    difft_t0 = t_t0[1:]-t_t0[:-1]
#    difft_t0 = difft_t0[np.where(difft_t0<period*1.2)[0]]
#    difft_t0 = difft_t0[np.where(difft_t0>period*0.8)[0]]
#    t_t0 = np.concatenate([[t_t0[0]-np.average(difft_t0)],t_t0,[t_t0[-1]+np.average(difft_t0),t_t0[-1]+np.average(difft_t0)*2]])#,[t_t0[-1]+avgdiff]])
#    difft_t0 = t_t0[1:]-t_t0[:-1]
#    t0inds2 = t0inds[(difft_t0>period*0.8)&(difft_t0<period*1.2)]
#    t_t0 = t_t0[(difft_t0>period*0.8)&(difft_t0<period*1.2)]
#    difft_t0 = t_t0[1:]-t_t0[:-1]
#    return difft_t0,t_t0,t0inds2
def Makett_t0(t_t0,raw_t):
    difft_t0 = t_t0[1:]-t_t0[:-1]
    tt = np.insert(difft_t0,np.arange(0,len(difft_t0),1),np.zeros(len(difft_t0)))  # a zero before every difference value
    tt = np.concatenate([tt,[0,np.average(difft_t0)]]) # put another set of 0,difference at the end of the array
    tt0 = np.insert(t_t0,np.arange(1,len(t_t0)+1,1),t_t0) # duplicate each t_t0 value so array is [a,a,b,b,c,c,...,z,z]
    tt0[:-1] = tt0[1:] # shift array to left by one index, this makes the last 3 indexes the same value: [a,b,b,c,c,...,z,z,z]
    tt0[-1] = tt0[-1] + np.average(difft_t0) # make last value one pulse width larger in time: [a,b,b,c,c,...,z,z,q]
    #tmt0 = np.interp(raw_t,tt0,tt)
    tt_t0 = np.array([tt0,tt])
    return tt_t0#,tmt0
def MakeCov1d(E,raw_t,tstepforcoverage,t_BSl,BSbinedges,lowang,highang,omega):
    tcov = np.arange(raw_t[0],raw_t[-1],tstepforcoverage)
    trot = np.average(t_BSl[1:]-t_BSl[:-1]);
    BSind=(t_BSl-raw_t[0])/tstepforcoverage; BSind=np.round(BSind); BSind = BSind.astype(np.int);
    BSc = bsa.fbs(tcov,t_BSl,BSind[1:-1],omega,lowang,highang,trot)
    E[23] = np.histogram(BSc,bins = BSbinedges)[0]*1.0
    sumcov = np.sum(E[23])
    print('sum of coverage = '+str(sumcov))
    E[23] = np.divide(E[23],sumcov)
    print('sum of coverage after reducing = '+str(np.sum(E[23])))
    return E
def MakeCov2d(E,raw_t,tstepforcoverage,t_t0,t_BSl,tbinedges,BSbinedges,lowang,highang,omega):
    tcov = np.arange(raw_t[0],raw_t[-1],tstepforcoverage)
    period = np.average(t_t0[1:]-t_t0[:-1]); trot = np.average(t_BSl[1:]-t_BSl[:-1]);
    tind=(t_t0-raw_t[0])/tstepforcoverage; tind=np.round(tind); tind = tind.astype(np.int);
    BSind=(t_BSl-raw_t[0])/tstepforcoverage; BSind=np.round(BSind); BSind = BSind.astype(np.int);
    tc = bsa.ft(tcov,t_t0,tind[1:-1],period); BSc = bsa.fbs(tcov,t_BSl,BSind[1:-1],omega,lowang,highang,trot)
    E[23] = np.histogram2d(tc,BSc,bins = [tbinedges,BSbinedges])[0]*1.0
    sumcov = np.sum(E[23])
    print('sum of coverage = '+str(sumcov))
    E[23] = np.divide(E[23],sumcov)
    print('sum of coverage after reducing = '+str(np.sum(E[23])))
    return E
def MakeCovold04202020(E,raw_t,tstepforcoverage,tbins,A3bins,tt_t0,t_BS,tbinedges,BSbinedges):
    tcov = np.arange(raw_t[0],raw_t[-1],tstepforcoverage)
    tc = np.interp(tcov,tt_t0[0],tt_t0[1])
    BSc = np.interp(tcov,t_BS[0],t_BS[1])
    E[23] = np.histogram2d(tc,BSc,bins = [tbinedges,BSbinedges])[0]*1.0
    sumcov = np.sum(E[23])
    print('sum of coverage = '+str(sumcov))
    E[23] = np.divide(E[23],sumcov)
    print('sum of coverage after reducing = '+str(np.sum(E[23])))
    return E
def normalize(E,raw_t,tstepforcoverage,tt,tt0,t_A3,tbinedges,A3binedges,tbins,A3bins):
    print('events in all channels before normalization = ' + str(np.sum(E[:20])))
    counts = 0
    x = int(tbins/3.0);y=int(A3bins/3.0)
    for i in range(20):
        counts = counts+np.sum(E[i][x:-x,y:-y])
    print('events mid 1/9 of all channels before normalization = ' + str(counts))
    #tstepforcoverage = pixeltime/2000
    tmt0coverage = np.interp(np.arange(raw_t[0],raw_t[-1],tstepforcoverage),tt0,tt)
    A3coverage = np.interp(np.arange(raw_t[0],raw_t[-1],tstepforcoverage),t_A3[0],t_A3[1])
    E[23] = np.histogram2d(tmt0coverage,A3coverage,bins = [tbinedges,A3binedges])[0]*1.0#*tstepforcoverage
    a = E[23][np.where(E[23]>0.0)]
    avgcov = np.average(a)             # check into this
    E[23] = np.divide(E[23],avgcov)
    
    nE = np.zeros(np.shape(E))
    for i in range(20):
        nE[i] = np.divide(E[i],E[23])
    aa = nE[:20][~np.isnan(nE[:20])]
    bb = aa[~np.isinf(aa)]
    print('events in all normalized channels = ' + str(np.sum(bb)))
    counts = 0
    x = int(tbins/3.0);y=int(A3bins/3.0)
    for i in range(20):
        cc = nE[i][x:-x,y:-y]
        aa = cc[~np.isnan(cc)]
        bb = aa[~np.isinf(aa)]
        counts = counts+np.sum(bb)
    print('events mid 1/9 of all channels after normalization = ' + str(counts))
    return nE
def notvisitedtonan(E,raw_t,tstepforcoverage,tt,tt0,t_A3,tbinedges,A3binedges,tbins):
    print('events in all channels before normalization = ' + str(np.sum(E[:20])))
    #tstepforcoverage = pixeltime/2000
    tmt0coverage = np.interp(np.arange(raw_t[0],raw_t[-1],tstepforcoverage),tt0,tt)
    A3coverage = np.interp(np.arange(raw_t[0],raw_t[-1],tstepforcoverage),t_A3[0],t_A3[1])
    E[23] = np.histogram2d(tmt0coverage,A3coverage,bins = [tbinedges,A3binedges])[0]*1.0#*tstepforcoverage
    nE = np.copy(E)
    for i in range(20):
        nE[i][np.where(E[23]==0)] = np.nan
    aa = nE[:20][~np.isnan(nE[:20])]
    bb = aa[~np.isinf(aa)]
   # E[23][np.where(E[23]
    print('events in all normalized channels = ' + str(np.sum(bb)))
    return nE
def plothist(h=[[]],title='',aspect=1,clim='auto'):
    plt.rcParams['axes.facecolor']='purple'
    fig = plt.figure()
    ax = fig.add_axes([0,0,1.4,1.2])
    im = ax.imshow(h,cmap='rainbow',aspect = aspect,interpolation='none',origin='low')  
    ax.set_xlabel('BS',fontsize = 20)
    ax.set_ylabel('t',fontsize = 20)
    #ax.set_axis(fontsize = 20)
    ax.set_title(title)
    cbar = fig.colorbar(im,shrink = 0.8)
    if clim != 'auto':
        im.set_clim(clim[0],clim[1])
    plt.show()
    print('number of zeros = '+str(len(np.where(h.flatten()==0)[0])))
    print('number of NaNs = '+str(np.sum(np.isnan(h.flatten()))))
def printdiag(nE,E,nn):
    a = nE[nn][:20][~np.isnan(nE[nn][:20])]
    a = a[~np.isinf(a)]
    b = E[nn][:20][~np.isnan(E[nn][:20])]
    b = b[~np.isinf(b)]
    print('normalized:')
    print('nans = '+str(np.sum(np.isnan(nE[nn][:20].flatten()))))
    print('infs = '+str(np.sum(np.isinf(nE[nn][:20].flatten()))))
    print('min = '+str(np.min(a)))
    print('max = '+str(np.max(a)))
    print('')
    print('not normalized')
    print('# of pixels = '+str(len(E[nn][:20].flatten()==0)))
    print('zeros = '+str(len(np.where(E[nn][:20].flatten()==0)[0])))
    print('nonzeros = '+str(len(np.where(E[nn][:20].flatten()!=0)[0])))
    print('nans = '+str(np.sum(np.isnan(E[nn][:20].flatten()))))
    print('infs = '+str(np.sum(np.isinf(E[nn][:20].flatten()))))
    print('min = '+str(np.min(b)))
    print('max = '+str(np.max(b)))
def unpackevts(filename):
    if filename[-4:] == '.evt':
        file = open(filename, mode = 'rb')
        filecontent = file.read()
        values = struct.unpack('2i2h'*int(len(filecontent)/12),filecontent) 
        values = np.array(values,dtype=float)
        valuescell = {0:values}
        files = [filename]
    else:
        files = glob.glob(filename + '/*.evt')
        files = np.sort(files)
        valuescell = {}
        for i in range(len(files)):
            file = open(files[i], mode = 'rb')
            filecontent = file.read()
            valuescell[i] = struct.unpack('2i2h'*int(len(filecontent)/12),filecontent) 
            valuescell[i] = np.array(valuescell[i],dtype=float)
    return files,valuescell
def Unpackvaluescell(valuescell,tstepforcoverage):
    timesind = np.arange(0,int(len(valuescell))-3,4)
    timensind = np.arange(1,int(len(valuescell))-2,4)
    IDind = np.arange(2,int(len(valuescell))-1,4)
    times = valuescell[timesind]
    timens = valuescell[timensind]
    ID = valuescell[IDind]
    nzeros = len(np.where(times==0)[0])*1.0
    ntotal = len(times)
    if len(np.where(times==0)[0]!=0):
        zeroindex = np.where(times!=0)[0]
        ID = ID[zeroindex]
        timens = timens[zeroindex]
        times = times[zeroindex]
    sortedtimes_index = times.argsort(); times = times[sortedtimes_index]; timens = timens[sortedtimes_index]; ID = ID[sortedtimes_index]
    times = times - times[0];  raw_t = times+timens*10**(-9)
    sortind = raw_t.argsort(); ID = ID[sortind]; raw_t = raw_t[sortind]
    tcov = np.arange(raw_t[0],raw_t[-1],tstepforcoverage); chh = np.zeros(len(tcov))+50; raw_t = np.concatenate([raw_t,tcov]); ID = np.concatenate([ID,chh]); 
    sumcov = len(tcov);
    sortind = raw_t.argsort(); ID = ID[sortind]; raw_t = raw_t[sortind]
    inds = {}
    for i in np.unique(ID):
        inds[i] = np.where(ID==i)[0]
    print('length of raw_t = ' + str(len(raw_t)))
    return raw_t,ID,inds,sumcov
