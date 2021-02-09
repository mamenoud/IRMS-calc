import numpy as np

# function for mixing ratio calculation
def MR(peakarea, samplevol, refarea, err_refarea, refvol, refID):
    
    if refID=='NAT336':
        refMR=1975.5
    elif refID=='CB09285 old':
        refMR=1974.0
    elif refID=='CB09285 new':
        refMR=1970.0
    elif refID=='JASKIER':
        refMR=1950.3
    elif refID=='COC':
        refMR=2528.3
    elif refID=='NAT350':
        refMR=1907
        
    n=len(peakarea)
    
    if type(refarea)==float:              # Make an array with the ref peak area value, in case it was not interpolated
        refarea=np.full((n), refarea)
        
    sampleMR=[]
    err_refMR=[]
    for i in range(n):
        calcMR=peakarea[i]*(refMR/refarea[i])*refvol/samplevol[i]
        sampleMR.append(calcMR)
        err_refMR.append(refMR*err_refarea/refarea[i])     
        
        i=i+1

    return sampleMR, err_refMR


#function for isotopic signature calculations
def iso3(rawsig, refsig, isotopID, refID):
    
    if refID=='NAT336':
        stdsig13C=-48.14
        stdsigD=-90.81
    if refID=='CB09285 old':
        stdsig13C=-47.75
        stdsigD=-87.93
    if refID=='CB09285 new':
        stdsig13C=-48.07
        stdsigD=-88.31
    if refID=='JASKIER':
        stdsig13C=-47.82
        stdsigD=-85.50
    if refID=='COC':
        stdsig13C=-51.72
        stdsigD=-136.76
    if refID=='NAT350':
        stdsig13C=-47.75
        stdsigD=-83.35
        
    if isotopID==1:
        stdsig=stdsig13C
    elif isotopID==2:
        stdsig=stdsigD
        
    n=len(rawsig)
    
    if (type(refsig)==float) or (type(refsig)==np.float64):               # Make an array with the ref sig value, in case it was not interpolated
        refsig=np.full((n), refsig)
        
    samplesig=[]
    i=0
    for i in range(len(rawsig)):
        calcdiff=((rawsig[i]-refsig[i])/(refsig[i]+1000))*1000
        calcsig=calcdiff+stdsig+(calcdiff*stdsig)/1000
        samplesig.append(calcsig)
        i=i+1
    
    return samplesig
