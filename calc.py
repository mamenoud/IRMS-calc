import numpy as np
import scipy as sc

# function for mixing ratio calculation
def MR(peakarea, samplevol, refarea, err_refarea, refvol, refID):
    
    if refID=='NAT336':
        refMR=1975.5
    elif refID=='CB09285':
        refMR=1974.0
    elif refID=='JASKIER':
        refMR=1950.3
    elif refID=='COC':
        refMR=2528.3
    elif refID=='NAT350':
        refMR=1907
        
    n=len(peakarea)
    
    sampleMR=[]
    err_refMR=[]
    i=0
    
    if type(refarea)==float:                            # Make an array with the ref peak area, in case it was not interpolated
        refarea=np.full((n), refarea)
        
    for i in range(n):
    
        calcMR=peakarea[i]*(refMR/refarea[i])*refvol/samplevol[i]
        sampleMR.append(calcMR)
                
    #error of the refs:
        err_refMR.append(refMR*err_refarea/refarea[i])      # If it was not interpolated, the created vector is all the same values
        
        i=i+1

    
    return sampleMR, err_refMR


#function for isotopic signature calculations
#def iso(rawsig, refsig, isotopID):
#    stdsig13C=-48.14
#    stdsigD=-90.81
#    if isotopID==1:
#        stdsig=stdsig13C
#    elif isotopID==2:
#        stdsig=stdsigD
#    samplediff=[]
#    samplesig=[]
#    i=0
#    for i in range(len(rawsig)):
#        calcdiff=((rawsig[i]-refsig)/(refsig+1000))*1000
#        samplediff.append(calcdiff)
#        calcsig=calcdiff+stdsig+(calcdiff*stdsig)/1000
#        samplesig.append(calcsig)
#        i=i+1
#    return samplediff, samplesig


#function for isotopic signature calculations
def iso3(rawsig, refsig, isotopID, refID):
    
    if refID=='NAT336':
        stdsig13C=-48.14
        stdsigD=-90.81
    if refID=='CB09285':
        stdsig13C=-47.75
        stdsigD=-87.93
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
    
    if type(refsig)==float:                            # Make an array with the ref sigs, in case it was not interpolated
        refsig=np.full((n), refsig)
        
    samplesig=[]
    i=0
    for i in range(len(rawsig)):
        calcdiff=((rawsig[i]-refsig[i])/(refsig[i]+1000))*1000
        calcsig=calcdiff+stdsig+(calcdiff*stdsig)/1000
        samplesig.append(calcsig)
        i=i+1
    
    return samplesig


def my_interpol(x_ref, y_ref, x_interpol):
            
    slope, intercept, r_value, p_value, std_slope = sc.stats.linregress(x_ref, y_ref)       # interpolation for getting the reference value for each sample in the set
            
    y_interpol = slope * x_interpol + intercept
            
    #sx2 = (x_ref**2).sum()
    #std_intercept = std_slope * np.sqrt(sx2/len(x_ref))
    #err_y = np.sqrt((x_interpol * std_slope)**2 + std_intercept**2)          # propagation of error: (sd_f(x,y))^2 = (df/dx * sd_x)^2 + (df/dy * sd_y)^2
    
    yp = x_ref*slope+intercept
    ms = np.sqrt(sum((y_ref - yp)**2)/len(x_ref))
    err_y = ms
    
    return y_interpol, err_y