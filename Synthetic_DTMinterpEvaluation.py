#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:37:54 2020

@author: zhangtianqi
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
import os
from sklearn import metrics

import sys
sys.path.append("D:/Workfolder_Zhang/myCodes/step_functions/attached_functions/DTMextraction")
import postProcessing

# canpStr = 'denseCanopy'
canpStr = 'moderateCanopy'
# canpStr = 'sparseCanopy'

# Topostr = 'mildTopo'
# Topostr = 'moderateTopo'
Topostr = 'steepTopo'

strings = ['D:/Workfolder_Zhang/myCodes/step_functions/synthetic_topography','/',
           canpStr+'2/']
path = ''.join(strings)
os.chdir(path)

path1 = 'D:/Workfolder_Zhang/myCodes/step_functions/synthetic_topography'
path2 = '/InterpResult3/'
path4 = '/model_evaluation/forSubmisson/'

csv_str = '_'+canpStr+'.csv'
img_str = canpStr+'.png'
mask_str = canpStr
groundmask = canpStr+'_mask'

modelEvaluate_str = '_'+Topostr+canpStr
DTM = io.imread(path1+'/'+Topostr+'Update.tif')

#%%
Nrow = np.shape(DTM)[0]
Ncol = np.shape(DTM)[1]
Nval = 30
mask = []
for i in range(0,Nval):
    Mask = mask_str+str(i+1)
    mask.append(Mask)
    
CanpMask =  np.zeros((Nrow,Ncol,Nval))
predFRK =  np.zeros((Nrow,Ncol,Nval))
predIDW =  np.zeros((Nrow,Ncol,Nval))
predInpain =  np.zeros((Nrow,Ncol,Nval))
predLin =  np.zeros((Nrow,Ncol,Nval))
predNatur =  np.zeros((Nrow,Ncol,Nval))
predCubic =  np.zeros((Nrow,Ncol,Nval))

for i in range(1,Nval+1): 
    CanpMask[:,:,i-1] = io.imread(groundmask+str(i)+'.tif')
    predFRK[:,:,i-1] = io.imread(path+path2+'OKInterp_'+Topostr+str(i)+'.tif')
    predIDW[:,:,i-1] = io.imread(path+path2+'IDWInterp_'+Topostr+str(i)+'.tif')
    predInpain[:,:,i-1] = io.imread(path+path2+'inpaintInterp_'+Topostr+str(i)+'.tif')
    predLin[:,:,i-1] = io.imread(path+path2+'linearInterp_'+Topostr+str(i)+'.tif')
    predNatur[:,:,i-1] = io.imread(path+path2+'naturalInterp_'+Topostr+str(i)+'.tif')
    predCubic[:,:,i-1] = io.imread(path+path2+'cubicInterp'+Topostr+str(i)+'.tif')

#%%
# canopyRatio = np.loadtxt(path1+canpStr+'2/'+canpStr+'Ratio.txt',delimiter =',')
# print(canopyRatio)

# #%%
# plt.imshow(CanpMask[:,:,5])
# plt.grid(False)
# plt.xticks([])
# plt.yticks([]) 
# plt.savefig('/Users/zhangtianqi/Documents/manuscript/DTM_extraction_from_ArcticDEM/figures_new_sites/'+
#             canpStr+'Mask.png',dpi = 200)

#%% visualize the interpolation
# numMask = 11
# print(canopyRatio[numMask])
# FRK = predFRK[:,:,numMask]
# IDW = predIDW[:,:,numMask]
# INPAINT = predInpain[:,:,numMask]
# BILIN = predLin[:,:,numMask]
# NN = predNatur[:,:,numMask]
# canopyMask = CanpMask[:,:,numMask]
# print(canopyRatio)

# import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep

# predictors = [DTM,FRK,IDW,BILIN,INPAINT,NN,canopyMask]
# title = ['DTM','FRK','IDW','BILIN','INPAINT','NN',
#           'canopy cover: '+str(round(canopyRatio[numMask],3))]

# fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(14,3))
# fig.subplots_adjust(wspace=0.05,hspace=0.1)
# for i, (ax, feature,title) in enumerate(zip(axes.flatten(), predictors,title)):
#     if i < 6:
#         hillshade_azimuth_210 = es.hillshade(feature, azimuth=210)
#         im=ep.plot_bands(
#         feature,
#         ax=ax,
#         cmap="terrain",
#         vmin=np.floor(np.nanmin(DTM)),
#         vmax=np.ceil(np.nanmax(DTM)),
#         cbar=False,
#         title = title
#         )
#         ax.imshow(hillshade_azimuth_210, cmap="Greys", alpha=0.5)
#     else:
#         ax.imshow(feature)
#         ax.set_title(title)
#     ax.grid(False)
#     ax.axis('on')
#     ax.set_xticks([])
#     ax.set_yticks([])   
 
# plt.savefig('/Users/zhangtianqi/Documents/manuscript/DTM_extraction_from_ArcticDEM/figures_new_sites/'+
#             'moderate_canopyCover_'+str(round(canopyRatio[numMask],3))+'.png',dpi = 200)
    
# plt.imshow(predFRK[:,:,0])    

#%% ground pixel index
flag_nan = np.isnan(predLin)
predFRK = np.where(flag_nan,np.nan,predFRK)
predIDW = np.where(flag_nan,np.nan,predIDW)
predInpain = np.where(flag_nan,np.nan,predInpain)
predLin = np.where(flag_nan,np.nan,predLin)
predNatur = np.where(flag_nan,np.nan,predNatur)
predCubic = np.where(flag_nan,np.nan,predCubic)

prediction = [predFRK,predIDW,predInpain,predLin,predNatur,predCubic]

RMSE = np.zeros((len(prediction),Nval))
rRMSE = np.zeros((len(prediction),Nval))
MAE = np.zeros((len(prediction),Nval))
MBE = np.zeros((len(prediction),Nval))
# R_sqrd = np.zeros((len(prediction),Nval))

for i in range(0,len(prediction)):
    for k in range(0,Nval):
        [RMSE[i,k],rRMSE[i,k],MAE[i,k],MBE[i,k]] = postProcessing.comparison_all(prediction[i][:,:,k],DTM)
        # Corr[i,k] = postProcessing.corrA(prediction[i][:,:,k],DTM)
        # R_sqrd[i,k] = postProcessing.calRsquared(prediction[i][:,:,k],DTM)
        
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]
# print(''.join(namestr(MAE,globals())))

def tableWrite(metric):
    Table = pd.DataFrame(metric, columns = mask,
                         index = ['OK','IDW','INPAINT','Linear','NN','Cubic']) 
    Mean = np.round(np.mean(metric, axis=1)*100,1)
    Std = np.round(np.std(metric, axis=1)*100,1)
    Min = np.round(np.min(metric, axis=1)*100,1)
    Max = np.round(np.max(metric, axis=1)*100,1)
    Table[''.join(namestr(metric,globals()))+'_mean'] = Mean
    Table[''.join(namestr(metric,globals()))+'_std'] = Std
    Table[''.join(namestr(metric,globals()))+'_min'] = Min
    Table[''.join(namestr(metric,globals()))+'_max'] = Max

    Str = []
    for i in range(0,len(Mean)):
        test = str(Mean[i,])+'±'+ str(Std[i,])
        Str.append([test])
    Table[''.join(namestr(metric,globals()))+modelEvaluate_str] = Str
    # Table.to_csv(path1+path4+''.join(namestr(metric,globals()))+'_IDW_'+Topostr+csv_str)
    return(Table)

RMSE_table = tableWrite(RMSE)
rRMSE_table = tableWrite(rRMSE)
MAE_table = tableWrite(MAE)
MBE_table = tableWrite(MBE)

RMSE = RMSE_table.loc[ : , ['RMSE_mean', 'RMSE_std','RMSE_min', 'RMSE_max',] ]
rRMSE = rRMSE_table.loc[ : , ['rRMSE_mean', 'rRMSE_std','rRMSE_min', 'rRMSE_max'] ]
# MAE = MAE_table.loc[ : , ['MAE_mean', 'MAE_std','MAE_min', 'MAE_max'] ]
# MBE = MBE_table.loc[ : , ['MBE_mean', 'MBE_std','MBE_min', 'MBE_max'] ]

#%%
def concateRMSE (metric):
    RMSEmeanStdl = metric.to_numpy()
    RMSE_TEST = np.empty((len(metric),0)).tolist()
    for i in range(0,len(RMSE_TEST)):
        # RMSE_TEST[i] = str(RMSEmeanStdl[i,0]) +' (' + str(RMSEmeanStdl[i,1]) + ")"+'[' + str(RMSEmeanStdl[i,2]) + ', '+ str(RMSEmeanStdl[i,3])+ "]"
        RMSE_TEST[i] = str(RMSEmeanStdl[i,0]) +' (' + str(RMSEmeanStdl[i,1]) + ")"
    metric[''.join(namestr(metric,globals()))] = RMSE_TEST 
    return(metric)

RMSE = concateRMSE (RMSE)
rRMSE = concateRMSE (rRMSE)
# MAE = concateRMSE (MAE)
# MBE = concateRMSE (MBE)

Table = RMSE.loc[:,['RMSE']].merge(rRMSE.loc[:,['rRMSE']],left_index=True, right_index=True)
# Table = Table.merge(MAE.loc[:,['MAE']],left_index=True, right_index=True)
# Table = Table.merge(MBE.loc[:,['MBE']],left_index=True, right_index=True)
Table.to_csv(path1+path4+'model_evaluation_meanStd_'+Topostr+csv_str)
print(canpStr+Topostr+':\n',Table)    
    
# #%%


    
    
    
    
    