import numpy as np

def c(t,omega,lowang,highang,trot):
    trot = trot*1.0
    alpha = omega**2/(omega*trot/2-(highang-lowang)); ta = omega/alpha; BSa=alpha*ta**2/2; BSout = np.zeros(len(t));
    ind = {}; ts  = {}; ts[0] = 0; ts[1] = ta; ts[2] = trot/2-ta; ts[3] = trot/2+ta; ts[4] = trot-ta; ts[5] = trot; # boundary times for piecewise BST(t) function
    BSout = np.concatenate([alpha*t[t<ts[1]]**2/2+lowang, omega*(t[(t>=ts[1])&(t<ts[2])]-ta)+BSa+lowang, -alpha*(t[(t>=ts[2])&(t<ts[3])]-trot/2)**2/2+highang, -omega*(t[(t>=ts[3])&(t<ts[4])]-(trot/2+ta))-BSa+highang, alpha*(t[t>=ts[4]]-trot)**2/2+lowang])
    return BSout#,[t1,t2,t3,t4,t5],[ta,tb,tc,td,te],alpha
def fbs(t,tBSl,BSlinds,omega,lowang,highang,trot):
    tper = np.zeros(len(t))
    BSout = np.zeros(len(t))
    ind0 = BSlinds[0]; inde = BSlinds[-1];
    tper[:ind0] = t[:ind0] - tBSl[0]              # taking care of first and last set of times
    tper[inde:] = t[inde:] - tBSl[-2]
    BSout[:ind0] = c(tper[:ind0],omega,lowang,highang,trot)
    BSout[inde:] = c(tper[inde:],omega,lowang,highang,trot)
    for i in range(len(BSlinds)-1):
        n = BSlinds[i]; m = BSlinds[i+1];
        tper[n:m] = t[n:m]-tBSl[i+1]
        BSout[n:m] = c(tper[n:m],omega,lowang,highang,trot)
    return BSout#c(tper,omega,lowang,highang,trot)
def ft(t,t0,t0inds,period): # making t minus t0 
    tper = np.zeros(len(t))
    ind0 = t0inds[0]; inde = t0inds[-1];
    tper[:ind0] = t[:ind0] - t0[0]              # taking care of first and last set of times
    tper[inde:] = t[inde:] - t0[-2]
    for i in range(len(t0inds)-1):
        n = t0inds[i]; m = t0inds[i+1];
        tper[n:m] = t[n:m]-t0[i+1]
    return tper