#import statistics
import pandas as pd

def avg(MR, sig, samples, volumes):
    #avgMR=[]
    #avgsig=[]
    #errMR=[]
    #errsig=[]
    
    df=pd.DataFrame({'ID':samples, 'MR':MR, 'sig':sig, 'vol':volumes})
    
    avgMR=df.groupby(['ID'])['MR'].mean()
    avgsig=df.groupby(['ID'])['sig'].mean()
    
    errMR=df.groupby(['ID'])['MR'].std()
    errsig=df.groupby(['ID'])['sig'].std()
    
    n=df.groupby(['ID']).size()
    
    vol=df.groupby(['ID'])['vol'].mean()
    
    #for sp in samples:
    #    spMR=MR[sp*rep:rep*sp+rep]
    #    spsig=sig[sp*rep:rep*sp+rep]
    #    avgMR.append(statistics.mean(spMR))
    #    avgsig.append(statistics.mean(spsig))
    #    errMR.append(statistics.stdev(spMR))
    #    errsig.append(statistics.stdev(spsig))

    return avgMR, errMR, avgsig, errsig, n, vol


#def grubbs(size, alpha):
#    sig=alpha/size
#    df=size-2
#    t-crit=student(1-sig, df)
