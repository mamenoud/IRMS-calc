#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:58:37 2018

@author: malika
"""

# This script is for regular data from Cryotiger at UU; d13C OR dD separately
#""""""""""""""


# Importing the data

import datetime
now = datetime.datetime.now()
import shutil, os
import re
import csv
import math
import statistics
import sys
import openpyxl
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd
from pandas import ExcelWriter

# Separate file for the mixing ratio and isotopic signature calculations
#import calc
from calc import MR
from calc import iso3
from calc import my_interpol
import mystat
import testdata

plt.rcParams.update({'font.size': 10})

#*****************************************************************************
# Hypothesis to identify the methane peak
#*****************************************************************************

#ref_id='NAT336'   # What's written as Identifier in the file

#ref_id='ref'
ref_id='COC'
#ref_id='CB09285'
#ref_id='CB9285'

#strref='NAT336'   # Real name of the ref that was used
#strref='CB09285'
#strref='JASKIER'
strref='COC'

dateformat='%d/%m/%Y %H:%M:%S'
permil=' [\u2030]'
MRname='MR [ppb]'
d13C=' -d13C'
dD=' -dD'
delta_13C='d13C VPDB'
delta_D='dD SMOW'

min_peakarea=2
max_peakarea=15

min_peakwidth=25
max_peakwidth=70

min_sig13C=-70
max_sig13C=-20

min_sigD=-270
max_sigD=-50

peakstart_13C=300
peakend_13C=380

peakstart_D=200
peakend_D=260



#*****************************************************************************
# Structure of the result file
#*****************************************************************************

# MODIFY HERE
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


pathname=os.getcwd()

# Path of the csv file from Isodat
filepath='/Users/malika/surfdrive/Data/Krakow/Calibration/Input files'    

# Name of the file
# ATTENTION the file must be with ',' separator, and '.' decimal separator
filename='KR_dD_JASKIER_2018-09-15_edited'

# What did we measure?
# 1 for roof air 
# 2 for samples?
what=2

# Isotopologue that was measured
# 1 for carbon13
# 2 for deuterium
isotop=2

# Volume that is measured (in ml)
# 'y' if in column
# 'n' to set manually
vol_check='n'
vol=60

# Output file name:
outname='KR_dD_JASKIER_2018-09-15'
outloc='results/'+outname+'_out'

# To force interpolation
interpol_areas=0
interpol_sigs=0

#*****************************************************************************
# Reading the file and tests
#*****************************************************************************

# the following is copying the raw data file into the new file we are working with, and archiving the old one
sourced=os.path.join(filepath, filename+'.csv')
shutil.copy(sourced, pathname+'/raw')

rawdata=pd.read_csv('raw/'+filename+'.csv', encoding='utf-8-sig')

#%%
# TEST 1
# Check if the file is full

datanum=len(rawdata)

if datanum==0:
    print('The data file is empty!')
    print('Goodbye.')
    sys.exit()
else:
    print('The data is now accessible. It has '+str(datanum-1)+' entries.')
    #rawdata=pd.read_csv('raw/'+filename+'.csv', encoding='utf-8-sig')

# enter the column names

col_peak='Peak Nr'
col_an='Analysis'
col_id='Identifier 1'
col_peakarea='Area All'
col_peakwidth='Width'
col_peakstart='Start'
col_time='Time Code'
#col_vol='Identifier 2'
col_vol='Comment'

rawdata=rawdata.replace(ref_id, strref)

# Separation of d13C and dD data
if isotop==1:
    striso='13C'
elif isotop==2:
    striso='dD'
    
if what==1:
    col_iso='Identifier 2'
    col_vol='Comment'
    data=rawdata.loc[rawdata[col_iso]==striso]
else:
    data=rawdata

# If the volumes are not stated, put them in the right column

if vol_check=='n':
    data[col_vol]=vol
    
# get the methane peak data only

methdata=[]

if isotop==1:
    methdata=data.loc[(data[col_peakstart]>peakstart_13C) & (data[col_peakstart]<peakend_13C)]
    lim_isotop=[min_sig13C, max_sig13C]
    col_sig='d 13C/12C'
    delta='d13C'
    strdelta='d13C VPDB'

elif isotop==2:
    methdata=data.loc[(data[col_peakstart]>peakstart_D) & (data[col_peakstart]<peakend_D)]
    lim_isotop=[min_sigD, max_sigD]
    col_sig='d 2H/1H'
    delta='dD'
    strdelta='dD SMOW'

methdata.dropna(how='any', subset=[col_sig], inplace=True)

# Show the peak numbers
print('Methane data was extracted for a number of '+str(len(methdata))+' measures.')
print('Peak numbers: ')
print(str(methdata[col_peak]))

# Tests on the raw data: peak area, peak width and raw isotopic signature

# Limit values for tests [min, max]
lim_peakarea=[min_peakarea, max_peakarea]
lim_peakwidth=[min_peakwidth, max_peakwidth]

testdata.test2(methdata, col_peakarea, lim_peakarea, 'peak area')

testdata.test2(methdata, col_peakwidth, lim_peakwidth, 'peak width')

testdata.test2(methdata, col_sig, lim_isotop, delta+' isotopic signature')


# Now we'll extract the samples rows and the ref rows, based on the Identifier 1 (name of the reference gas)

def separe_sp_ref(meth_data):
    
    meth_data=meth_data.reset_index(drop=True)
    
    ref_data=meth_data.loc[meth_data[col_id]==strref]
    ref_data=ref_data.reset_index(drop=True)
    
    while meth_data[col_an][0]<ref_data[col_an][0]:                   # check if the first analysis is a ref
        meth_data=meth_data.drop(meth_data.index[0])
        meth_data=meth_data.reset_index(drop=True)
        
    while (meth_data[col_an].iloc[-1]>ref_data[col_an].iloc[-1]):
        meth_data=meth_data.drop(meth_data.index[-1])
        meth_data=meth_data.reset_index(drop=True)
    
    sp_data=meth_data.loc[meth_data[col_id]!=strref]
    sp_data=sp_data.reset_index(drop=True)
    
    return meth_data, sp_data, ref_data
    
def convert_time(data):
    
    times = pd.to_datetime(data[col_time], 
                           #infer_datetime_format=True)
                           infer_datetime_format=True, dayfirst=True)
    data[col_time] = times
    
    return data
    
methdata, samples, refs = separe_sp_ref(methdata)   
methdata = convert_time(methdata)     
samples = convert_time(samples)                             
refs = convert_time(refs)


#fig_all, axes_all = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), sharex=True)
#ax3=axes_all[0]
#ax4=axes_all[1]


# Get the analysis numbers of the air and refs

indexref=refs[col_an]

print('Analysis numbers of the reference gas:\n')
print(delta+'\n'+str(indexref))

index=samples[col_an]

if samples.empty==False:
    print('Analysis numbers of the samples measurements: ')
    print(delta+'\n'+str(index))
elif samples.empty==True:
    print('\nNo samples were measured.\n')

#%%
# Test of the regularity of the references sets
    
ref_areas = refs[col_peakarea]
ref_sigs = refs[col_sig]
ref_analysis = refs[col_an]
    
alpha = 0.05
r_lim=0.25

# Compute the standard deviation of peak areas and raw signatures for the working standards
areas_s = np.std(ref_areas)
sigs_s = np.std(ref_sigs)
print('\n**********************\nDistribution statistics\nstd(peak area) = %0.2f' % areas_s+'\nstd(raw sigantures) = %0.2f' % sigs_s+'\n')

print('\n**********************\nShapiro-Wilk test\nalpha = '+str(alpha))
areas_p, areas_norm = testdata.normality(ref_areas, 'areas', alpha)
sigs_p, sigs_norm = testdata.normality(ref_sigs, 'signatures', alpha)
    
print('\n**********************\nCorrelation test\nr < '+str(r_lim))
areas_r, areas_corr = testdata.correlation(ref_areas, ref_analysis, 'areas', r_lim)
sigs_r, sigs_corr = testdata.correlation(ref_sigs, ref_analysis, 'signatures', r_lim)
    
# Plot an overview of the results
if samples.empty==False:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 7), sharex=True)
    ax1=axes[0,0]
    ax2=axes[1,0]
    ax3=axes[0,1]
    ax4=axes[1,1]
    refs.plot(y=col_peakarea, x=col_time, ax=ax1, style='o', mfc='orange', mec='orange')
    refs.plot(y=col_sig, x=col_time, ax=ax2, style='o', mfc='red', mec='red')

    refs.plot(y=col_peakarea, x=col_time, ax=ax3, style='o', mfc='orange', mec='orange')
    samples.plot(y=col_peakarea, x=col_time, ax=ax3, style='s', mfc='blue', mec='blue')
    refs.plot(y=col_sig, x=col_time, ax=ax4, style='o', mfc='red', mec='red')
    samples.plot(y=col_sig, x=col_time, ax=ax4, style='s', mfc='blue', mec='blue')
    
elif samples.empty==True:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,7), sharex=True)
    ax1=axes[0]
    ax2=axes[1]
    refs.plot(y=col_peakarea, x=col_time, ax=ax1, style='o', mfc='orange', mec='orange')
    refs.plot(y=col_sig, x=col_time, ax=ax2, style='o', mfc='red', mec='red')

    
fig.savefig(outloc+'.png', dpi=300)

    
if (areas_norm == 0) or (areas_corr == 1):
    print('\n**********************\nWarning: peak area data doesn''t look stable. The values will be interpolated.')
    ans=input()
    interpol_areas=1
    if ans=='no':
        interpol_areas=0
        print('\nThe peak areas won''t be interpolated. The reference will be the average between the previous and following working standards.\n')
if (sigs_norm == 0) or (sigs_corr == 1):
    print('**********************\nWarning: isotopic signature data doesn''t look stable. The values will be interpolated.')
    ans=input()
    interpol_sigs=1
    if ans=='no':
        interpol_sigs=0
        print('\nThe isotope values won''t be interpolated. The reference will be the average between the previous and following working standards.\n')


# summary of stability results for exporting in excel
stab_df = pd.DataFrame({'Stdev':[areas_s, sigs_s], 'ShapiroWilk p':[areas_p, sigs_p], 'Corr r2':[areas_r, sigs_r], 'Interpolation':[interpol_areas, interpol_sigs]},
                        columns=['Stdev', 'ShapiroWilk p', 'Corr r2', 'Interpolation'], index=[ col_peakarea, col_sig])

# Ok, now do these corrections!

# Get the samples volumes in a list 
# Check if the sample volume is in Comment and in mL


#%%
#*****************************************************************************
# Calculating the values and export them
#*****************************************************************************

MRname='MR [ppb]'
refsigname='delta_ref'

col_delta=strdelta

#%%

# Now we want to calculate the mixing ratio in each sample set

def processing (all_meth_data, samples, refs, iso):
    
    # sort the index from 0 for the specific data, and according to analysis number for the general data
    all_meth_data.sort_values(by=col_an, inplace=True)
    meth_data=all_meth_data.reset_index(drop=True)

    n=len(meth_data)
    
    print('\n********************\n'+delta+' - '+str(n)+' measurements\n********************')
        
    # Get the first reference analysis numbers of each set
    newref=[meth_data[col_an][0]]
    
    for i in range(n-1):
        if (meth_data[col_id][i+1]==strref) & (meth_data[col_id][i]!=strref):
            #(all_meth_data[col_id][j-1]!=strref) & (all_meth_data[col_iso][j-1]!=all_meth_data[col_iso][j])):
            newref.append(meth_data[col_an][i+1])
    
    numset=len(newref)-1
    print('\nNumber of sets:'+str(numset))
    
    all_MRs=[]
    all_signatures=[]
    all_err_refMR=[]
    all_err_refsig=[]
    n_ref=[]
    
    for k in range(numset):
        
        # condition for the last set
        if k==numset-1:
            set_ref_prev=refs.loc[(refs[col_an]>=newref[k]) & (refs[col_an]<newref[k+1])]
            set_ref_foll=refs.loc[(refs[col_an]>=newref[k+1])]
            
        else:
            set_ref_prev=refs.loc[(refs[col_an]>=newref[k]) & (refs[col_an]<newref[k+1])]            # select the set of ref before and after
            set_ref_foll=refs.loc[(refs[col_an]>=newref[k+1]) & (refs[col_an]<newref[k+2])] 
             
            
        # the set of ref will be all 3 previous and following
        set_ref=pd.concat([set_ref_prev, set_ref_foll])
        set_ref=set_ref.reset_index(drop=True)
        #set_ref.dropna(how='any', inplace=True)
        
        set_sp=samples.loc[(samples[col_an]>set_ref[col_an][0]) & (samples[col_an]<set_ref[col_an].iloc[-1])]  # select the set of samples
        set_sp=set_sp.reset_index(drop=True)
        #set_sp.dropna(how='any', inplace=True)
        
        time_ref=pd.to_numeric(set_ref[col_time], downcast='float')                   # convert times into numeric
        time_sp=pd.to_numeric(set_sp[col_time], downcast='float')
        
        if interpol_areas == 1:                                      # Case where we need to interpolate from the refs, because they are not stable
            
            peakarea_ref, err_peakarea_ref = my_interpol(time_ref, set_ref[col_peakarea], time_sp)
     
            #print('sx2 '+str(sx2)+'\nstd_intercept '+str(std_intercept)+'\nerr_peakarea_ref '+str(err_peakarea_ref))
                
        if interpol_sigs == 1:
            
            signature_ref, err_sigs_ref = my_interpol(time_ref, set_ref[col_sig], time_sp)
            
        # If no interpolation, we keep only the last 2 of the 3 refs in the previous and following sets to make the average
        if len(set_ref_prev)>2:
            set_ref_prev=set_ref_prev[-2:]
        if len(set_ref_foll)>2:
            set_ref_foll=set_ref_foll[-2:]
        if len(set_ref_prev)!=len(set_ref_foll):
            print('WARNING! Unbalanced number of references')
                
        vol_ref_prev=statistics.mean(set_ref_prev[col_vol])
        vol_ref_foll=statistics.mean(set_ref_foll[col_vol])
        volume_ref=statistics.mean([vol_ref_prev, vol_ref_foll])                 # get the reference volume
        
        peakareas=set_sp[col_peakarea]                                           # get the air/samples peak areas
        volumes=set_sp[col_vol]                                                  # get the air/samples volumes
        signatures=set_sp[col_sig] 
        
        if interpol_areas == 0:

            # the set of refs will be only the last 2 previous and the last 2 following
            set_ref=pd.concat([set_ref_prev, set_ref_foll])
            set_ref=set_ref.reset_index(drop=True)

            peak_ref_prev=statistics.mean(set_ref_prev[col_peakarea])
            peak_ref_foll=statistics.mean(set_ref_foll[col_peakarea])
                
            peakarea_ref=statistics.mean([peak_ref_prev, peak_ref_foll])         # get the reference peak area
            err_peakarea_ref=np.std([peak_ref_prev, peak_ref_foll])
            
        if interpol_sigs == 0:
            
            # the set of refs will be only the last 2 previous and the last 2 following
            set_ref=pd.concat([set_ref_prev, set_ref_foll])
            set_ref=set_ref.reset_index(drop=True)
            
            sig_ref_prev=statistics.mean(set_ref_prev[col_sig])
            sig_ref_foll=statistics.mean(set_ref_foll[col_sig])
            signature_ref=statistics.mean([sig_ref_prev, sig_ref_foll])          # get the reference raw signature
                      
            err_sigs_ref = np.std(set_ref[col_sig])
                                        
        # Calculation of the mixing ratios
        set_MR, set_err_refMR = MR(peakareas, volumes, peakarea_ref, err_peakarea_ref, volume_ref, strref)
        
        # Calculation of the standardised isotopic signatures
        set_sig = iso3(signatures, signature_ref,  iso, strref)
        
        for q in range(len(set_sp)):                                             # adding the new calculated value to a general serie
            all_MRs.append(set_MR[q])
            all_signatures.append(set_sig[q])
            all_err_refMR.append(set_err_refMR[q])
            all_err_refsig.append(err_sigs_ref)
            n_ref.append(len(set_ref))
    
        #str_set = 
        # Print things to know where you are
        print('\nSet #'+str(k+1)+':\nreferences\n'+str(set_ref[[col_time, col_an, col_id, col_peakarea, col_sig]])+'\nsamples\n'+str(set_sp[[col_time, col_an, col_id, col_peakarea, col_sig]]))
        
        
        if interpol_areas == 1:
             print('\nInterpolated ref peak area values:\n'+str(peakarea_ref)+'\nerror:\n'+str(err_peakarea_ref))
             
        if interpol_sigs == 1:
            print('\nInterpolated ref raw signatures values:\n'+str(signature_ref)+'\nerror:\n'+str(err_sigs_ref))
             
    result_df=pd.DataFrame({col_an:samples[col_an], col_time:samples[col_time], col_id:samples[col_id], col_vol:samples[col_vol], MRname: all_MRs, delta: all_signatures, 'n_refs': n_ref ,'err_refMR [ppb]': all_err_refMR, 'err_refdelta'+permil: all_err_refsig}, 
    columns=[col_an, col_time, col_id, col_vol, MRname, delta, 'n_refs', 'err_refMR [ppb]', 'err_refdelta'+permil])
              
    return newref, numset, result_df
 
refchange, numset, results = processing (methdata, samples, refs, isotop)

#%%

# Write everything in the excel file

date=now.strftime('%Y-%m-%d')
time=now.strftime('%H%M%S')
print('**********************\nWriting the results in an Excel file...')

# Open the result files
writer = ExcelWriter(outloc+'.xlsx', engine='xlsxwriter')

# Write the raw data in the first sheet
data.to_excel(writer, sheet_name='isodat', index=False)

# Write the methane data in the second sheet
methdata.to_excel(writer, sheet_name='raw methane', index=False)

# Write the stability results in the third sheet
stab_df.to_excel(writer, sheet_name='stability', float_format='%0.2f')
refs.to_excel(writer, sheet_name='stability', index=False, startrow=len(stab_df)+2, columns=[col_an, col_time, col_id, col_vol, col_peakarea, col_sig])

# Write the processed data in the fourth sheet
results.to_excel(writer, sheet_name='processed', index=False)

#wb = writer.book


#%%


#write_csv = csv.writer(out_csv, delimiter=',')

n_out=len(results)

#header=csv.DictWriter(out_csv)

#header=['Stability of working standard '+strref,
        #'std (peak area) = '+str(np.std(ref_areas))+'\nstd (raw signatures) = '+str(np.std(ref_sigs))]

#for l in range(header):
#    write_csv.writerow(l)
#write_csv.writerow(l1)
#header.writeheader(str('Stability of working standard '+strref))
#write_csv.writerow('shapiroWilk p (peak area) = '+str(areas_p)+'\nshapiroWilk p (raw signatures) = '+str(sigs_p))
#write_csv.writerow('Correlation r (peak area) = '+str(areas_r)+'\nCorrelation r (raw signatures) = '+str(sigs_r))

#if interpol_areas == 1:
#    write_csv.writerow('Linear interpolation of peak areas.')
#if interpol_sigs == 1:
#    write_csv.writerow('Linear interpolation of isotopic signatures.')

#write_csv.writerow('')
#write_csv.writerow([col_time, col_id, col_vol, MRname+delta, strdelta, 'n_refs', 'err_refMR [ppb]', 'err_refdelta'+permil])

#for i in range(n_out):
#    write_csv.writerow(results.iloc[i])
    
#%% If it was continuous air measurements, save the file now, and a copy in csv
    
if what==1:
    writer.save()
    
    # make a new result table for the csv export
    results_csv=pd.DataFrame({col_an:results[col_an], col_time:results[col_time], col_id:results[col_id], col_vol:results[col_vol], MRname: results[MRname],  delta: results[delta], 'n_refs': results['n_refs'], 'err_refMR [ppb]': results['err_refMR [ppb]'], 'err_refdelta'+permil: results['err_refdelta'+permil], 'iso':[striso]*len(results)}, 
    columns=[col_an, col_time, col_id, col_vol, MRname, delta, 'n_refs', 'err_refMR [ppb]', 'err_refdelta'+permil, 'iso'])
    
    # Open the csv result files
    out_csv = open(outloc+'.csv', 'w', newline='', encoding='utf-8-sig')
    write_csv = csv.writer(out_csv)

    # mettre les en-têtes: Sample ID, Volume, calc MR, relative sig, standard sig
    #totcol=8
    MRname='MR [ppb]'
    refsigname='delta_ref'
    
    n_out=len(results)

    write_csv.writerow([col_an, col_time, col_id, col_vol, MRname+' -'+delta, strdelta, 'n_refs', 'err_refMR [ppb]',  'err_refdelta'+permil, col_iso])
    for line in range(n_out):
        write_csv.writerow(results_csv.iloc[line])

    out_csv.close()
    
    
# Now we want to average the duplicate measurements in case of samples
    
elif what==2:                             # sample measurements
    
    names=samples[col_id]
    
    [avgMR, errMR, avgsig, errsig, n, avgV]=mystat.avg(results[MRname], results[delta], names, results[col_vol])
    
    n_tot=len(avgMR)
   
    summary_df=pd.DataFrame({col_id:avgMR.index, MRname+' -'+delta:avgMR,'err_'+MRname+' -'+delta:errMR, strdelta:avgsig, 'err_'+strdelta:errsig, 'n':n, 'vol [ml]':avgV},
                            columns=[col_id, MRname+' -'+delta, 'err_'+MRname+' -'+delta, strdelta, 'err_'+strdelta, 'n', 'vol [ml]'])
    
    summary_df.to_excel(writer, sheet_name='processed', index=False, startcol=0, startrow=n_out+2, float_format='%0.2f')
    
    # save the output file
    writer.save()


print('\n********************\nResults successfuly saved in file '+outloc+'.xslx\n********************')