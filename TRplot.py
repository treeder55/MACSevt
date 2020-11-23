import numpy as np
import scipy as sp
import glob
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from lmfit import models
from lmfit import Model


def trplot(x,y,index = [0],xlim = 'auto',ylim='auto',xlabel='',ylabel='',off = 0,labelcolor = 'black',plotsize=[2.1,2.5],markersize=2,linewidth=0.5,marker='o',linestyle='-',font1=30,font2=20,title=''):
    plt.rcParams['axes.facecolor']='white'
    if len(np.shape(x))==1:
        x = np.array([x])
        y = np.array([y])
    colors = cm.rainbow(np.linspace(0, 1, len(x)))
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    for i in index:
        ax.plot(x[i],y[i]+i*off,marker=marker,color=colors[i],markersize = markersize,linewidth=linewidth)
    ax.set_ylabel(ylabel, color=labelcolor,fontsize = font1)
    ax.set_xlabel(xlabel, color=labelcolor,fontsize = font1)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
    #ax.legend(fontsize = font2,markerscale = 5)
    ax.set_title(title,fontsize=font2,color=labelcolor)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])    
    if xlim != 'auto':
        ax.set_xlim([xlim[0],xlim[1]])
def plot2d(x,y,index = [0],xlim = [0,1.4],ylim='auto',off = 0,labelcolor = 'black',plotsize=[2.1,2.5]):
    colors = cm.rainbow(np.linspace(0, 1, len(x)))
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    for i in index:
        ax.plot(data[i][0],data[i][1]+i*off,'-o',color=colors[i],markersize = 2,linewidth=0.5,markeredgewidth = 1,label = str(fields[i])+' T, i = ' + str(i))
    ax.set_ylabel('I', color=labelcolor,fontsize = 30)
    ax.set_xlabel('E (meV)', color=labelcolor,fontsize = 30)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
    ax.legend(fontsize = 14,markerscale = 5)
    ax.set_xlim([xlim[0],xlim[1]])
    ax.set_title('offset = ' + str(off),fontsize=20,color=labelcolor)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])
def plothist(h=[[]],title='',aspect=1,index = [0],clim='auto',xlim = 'auto',ylim='auto',xlabel='',ylabel='',off = 0,labelcolor = 'black',plotsize=[2.1,2.5]):
    plt.rcParams['axes.facecolor']='white'
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    im = ax.imshow(h,cmap='rainbow',aspect = aspect,interpolation='none',origin='low')  
    ax.set_xlabel('BS',fontsize = 20)
    ax.set_ylabel('t',fontsize = 20)
    #ax.set_axis(fontsize = 20)
    ax.set_title(title)
    cbar = fig.colorbar(im,shrink = 0.8)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])    
    if xlim != 'auto':
        ax.set_xlim([xlim[0],xlim[1]])
    if clim != 'auto':
        im.set_clim(clim[0],clim[1])
    plt.show()
def trfit2G(x,y,hints):
    model = models.GaussianModel(prefix='one_')+models.GaussianModel(prefix='two_')
    parnames = model.param_names
    pars = model.make_params()
    for j,n in enumerate(parnames):
        pars[n].set(value = hints[j],vary=True)
    result = model.fit(y,pars,x=x)
    print(result.fit_report())
    return result
#def oplotsim(self,index,off):
#    for i in index:
#        self.ax.plot(self.x,self.sim[i]+i*off,'-',color=self.colors[i],markersize = 2,linewidth=0.5,markeredgewidth = 1,label = 'sim, i = ' + str(i))
#def oplotfit(self,index,off):
#    for i in index:
#        self.ax.plot(self.x,self.simfit[i]+i*off,'-',color=self.colors[i],markersize = 2,linewidth=0.5,markeredgewidth = 1,label = 'sim, i = ' + str(i))
