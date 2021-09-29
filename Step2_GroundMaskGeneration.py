#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:37:09 2020

@author: zhangtianqi
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import sys
sys.path.append("D:/Workfolder_Zhang/myCodes/step_functions/attached_functions/DTMextraction/")

import numpy as np
from skimage import io
import os
import rasterio
import os.path as osp
import postProcessing
import matplotlib.pyplot as plt
from skimage.morphology import square,erosion

path = "D:/Workfolder_Zhang/Data/DigitalTerrainModel/test_region_alaska/study_region3_new/" 
os.chdir(path)

SiteStr = 'SiteOne'
result_path = osp.join(path,SiteStr+'_results/')
if osp.exists(result_path) == False:
    os.makedirs(result_path)  
    
#%%
MP = io.imread(result_path+'MP.tif')
DSM = io.imread(result_path+'ArcticDEM.tif')

#-- GMM mask --
n_components = 4
maxProb =io.imread(result_path + 'maxProbab.tif')
labelimg = io.imread(result_path +'cluster.tif')

# import matplotlib.colors as clr
# from pylab import *

# plt.imshow(labelimg,cmap=cm.get_cmap('tab10',n_components))
# plt.colorbar(boundaries=np.linspace(0.5,0.5+n_components,n_components+1),
#               ticks=np.arange(n_components)+1)
# plt.title("clusters",fontsize=10)
# plt.axis('off')

Ground = (labelimg == 1)|(labelimg == 4) #SiteFour

#%% generate ground mask for ICP coregistration: (maxProb>0.9) & morphological operation
Prob = 0.9
groundMask = Ground&(maxProb>Prob)
groundMask = erosion(groundMask,square(3))
ground_mask_for_coregistration = groundMask&(~np.isnan(MP))
plt.imshow(ground_mask_for_coregistration)
np.savetxt(result_path+ 'ground_mask_for_coregistration.txt', ground_mask_for_coregistration)

#%% localized refinement on GMM mask
def groundMaskLocRefine(DSM,groundmask,maxProb,ws=5,gamma=0.1, theta=4):
    """   
    DSM: the unprocessed DEM
    groundmask: original binary mask, where 1 indicate the ground location and 0 represents the non-ground. 
    ws: local window size, e.g.,5
    gamma: percentage of remaining ground pixels within a local window, default = 0.1
    theta: standard deviation threshold of elevation, default = 4m

    """
    Nrow = groundmask.shape[0]
    Ncol = groundmask.shape[1]
    groundMask = np.copy(groundmask)
    
    maskProb8 = maxProb>0.8
    groundMask8 = groundMask & maskProb8
    groundMask8erosion = erosion(groundMask8,square(3))
    
    for i in range(0, Nrow-ws-1):
        for j in range(0, Ncol-ws-1):
            meshPtLoc = groundMask8[i:i+ws,j:j+ws]
            DSMloc = DSM[i:i+ws,j:j+ws]
            DSMloc_std = np.nanstd(DSMloc)
            if (DSMloc_std>theta) and (np.sum(meshPtLoc)/(ws**2) < gamma): 
                groundMask[i:i+ws,j:j+ws] = groundMask8[i:i+ws,j:j+ws]              
            else:
                groundMask[i:i+ws,j:j+ws] = groundMask8erosion[i:i+ws,j:j+ws]
                
    return groundMask8, groundMask8erosion, groundMask

groundMaskGMM8, ground_open8, groundMaskGMMloc = groundMaskLocRefine(DSM,Ground,maxProb)&(~np.isnan(MP))

# store ground masks
DSMrstr = rasterio.open(result_path+'ArcticDEM.tif')
predList = [Ground,groundMaskGMM8,ground_open8,groundMaskGMMloc]
predName = ['croppedgroundMaskGMM','croppedgroundMaskGMM8','croppedgroundMaskGMM8open','croppedgroundMaskGMMloc']

for i in np.arange(0,len(predList)):
    postProcessing.writeRaster(DSMrstr,np.float32(predList[i]),
                                predName[i],result_path)