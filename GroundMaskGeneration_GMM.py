#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:53:49 2020

@author: zhangtianqi
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from skimage import io
import os
import os.path as osp
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from skimage.morphology import square,erosion

path = 'D:/Workfolder_Zhang/Data/DigitalTerrainModel/test_region_alaska/study_region3_new/' # path where you put the data
os.chdir(path)

result_path = osp.join(path,'results/')
if osp.exists(result_path) == False:
    os.makedirs(result_path)   
    
#%% Feature preparation for GMM classification
def spectraIdxCal(MB):
    red = MB[:,:,4]
    nir = MB[:,:,6]
    green = MB[:,:,2]
    NDVI = (nir - red)/(nir + red)
    MSAVI = (2*nir + 1 - np.sqrt(np.square(2*nir + 1) - 8*(nir - red)))/2
    NDWI = (green - nir)/(green + nir) 
    out = np.dstack((NDVI,MSAVI,NDWI))
    return out  

MBm = io.imread(path +'MB.tif') 
spectraIDXm = spectraIdxCal(MBm)
img = np.dstack((spectraIDXm,MBm))

#%% MODEL SELECTION: find the optimal number of clusters
# from matplotlib.patches import Ellipse
from matplotlib import rcParams
rcParams['figure.figsize'] = 16, 8

def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]

# PCA dimension reduction and BIC selection
from sklearn.decomposition import PCA
def standardize(img):
    img_mean = np.nanmean(img)
    img_std = np.nanstd(img)
    result = (img - img_mean)/img_std
    return result

#-- PCA ---
def dimReduction(img,n_components=4):
    Nrow = np.shape(img)[0]
    Ncol = np.shape(img)[1]
    Nval = np.shape(img)[2]
    finalV = np.zeros((Nrow*Ncol,Nval))
    for i in np.arange(0,Nval):        
        finalV[:,i] = np.reshape(standardize(img[:,:,i]),(Nrow*Ncol,)) # standardize all input features to ensure they are on the same scale

    # input data stucture [#pixels x #features]
    pca = PCA(n_components=n_components)
    pca.fit(finalV)
    print(sum(pca.explained_variance_ratio_))
    data = pca.transform(finalV) 
    return data

Nrow = np.shape(img)[0]
Ncol = np.shape(img)[1]
n_componets = 3
data = dimReduction(img,n_components=n_componets) # reduce data dimension
dataM = data.reshape(Nrow,Ncol,n_componets)

#-- BIC ---
n_clusters=np.arange(2, 10)
bics=[]
bics_err=[]
iterations=5
for n in n_clusters:
    tmp_bic=[]
    for _ in range(iterations):
        gmm=GMM(n, n_init=2).fit(data) 
        tmp_bic.append(gmm.bic(data))
    val=np.mean(SelBest(np.array(tmp_bic), int(iterations/5)))
    err=np.std(tmp_bic)
    bics.append(val)
    bics_err.append(err)
 
np.savetxt(result_path+'BICscore'+'.txt',bics)
np.savetxt(result_path+'BICerr'+'.txt',bics_err)

# Plot BIC score
plt.subplots(1,2, figsize=(8,5), sharey=True)
plt.subplot(121)
plt.errorbar(n_clusters,bics, yerr=bics_err, label='BIC')
plt.title("BIC Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("Score")
plt.legend()

plt.subplot(122)
plt.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')
plt.title("Gradient of BIC Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("grad(BIC)")
plt.legend()

#%% GMM clustering based on the BIC results
n_components=4 # optimal number of cluster
clf = GMM(n_components=n_components, covariance_type='full')
clf.fit(data)
Y_ = clf.predict(data) # Y_:labels

## probability conditional on different classes
probab = clf.predict_proba(data)

### entropy
from scipy import stats
ie = stats.entropy(probab.T)
IEimg = np.reshape(ie,(Nrow,Ncol))

### probability corresponding to the highest score
maxProbab = np.max(probab,axis = 1)
maxProbab.shape
maxProbabimg = np.reshape(maxProbab,(Nrow,Ncol))

### clustering result
labelimg = np.reshape(Y_,(Nrow,Ncol))+1

#%% select the pseudo-ground cluster: NDWI<-0.1 and lowest NDVI
NDVI = img[:,:,0]
SAVI = img[:,:,1]
NDWI = img[:,:,2]

label1 = labelimg == 1
label2 = labelimg == 2
label3 = labelimg == 3
label4 = labelimg == 4

medianNDVI = [np.median(NDVI[label1]),np.median(NDVI[label2]),
              np.median(NDVI[label3]),np.median(NDVI[label4])] 
medianNDWI = [np.median(NDWI[label1]),np.median(NDWI[label2]),
              np.median(NDWI[label3]),np.median(NDWI[label4])]

print('NDVI:',medianNDVI)
print('NDWI:',medianNDWI)

#%% generate ground mask for ICP coregistration: (maxProb>0.9) & morphological operation
MP = io.imread(result_path+'MP.tif') # binary mask where 1 indicates the matched points in stereo images
DSM = io.imread(result_path+'ArcticDEM.tif')

Ground = (labelimg == 1)|(labelimg == 4) 
Prob = 0.9
groundMask = Ground&(maxProbab>Prob)
groundMask = erosion(groundMask,square(3))
ground_mask_for_coregistration = groundMask&(~np.isnan(MP))
# plt.imshow(ground_mask_for_coregistration)
np.savetxt(result_path+ 'ground_mask_for_coregistration.txt', ground_mask_for_coregistration)

#%% localized refinement on GMM mask
def groundMaskLocRefine(DSM,groundmask,maxProb,ws=5,gamma=0.1, theta=4):
    """   
    DSM: the unprocessed DEM
    groundmask: original binary mask, where 1 indicate the ground location and 0 represents the non-ground. 
    ws: local window size, e.g.,5
    gamma: percentage of remaining ground pixels within a local window, default = 0.1
    theta: standard deviation threshold of elevation, default = 4m
    
    return:
        groundMask8: ground mask with maximum probability >0.8
        ground_open8: ground mask with maximum probability >0.8 and morphological operation
        groundMask: ground mask with maximum probability >0.8 and a locally adjusted morphological operation

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

groundMaskGMM8, ground_open8, groundMaskGMMloc = groundMaskLocRefine(DSM,Ground,maxProbab)&(~np.isnan(MP))

#%% store the ground masks
import rasterio
def writeRaster(DSMrstr,tiffMatrix,file_str,result_path):
    # DSMrstr: rasterio object
    # tiffMatrix: matrix you want to save, be sure to make the file type same as reference rasterio object: DSMrstr
    # file_str: name of the saved tiff file, e.g.,'IDWInterp_predGMM'
    # result_path: directory where you want to store the result
    
    image = rasterio.open(result_path+file_str+'.tif','w',driver='Gtiff',
                              width=tiffMatrix.shape[1], 
                              height = tiffMatrix.shape[0], 
                              count=1, crs=DSMrstr.crs, 
                              transform=DSMrstr.transform, 
                              dtype=DSMrstr.dtypes[0])
    image.write(tiffMatrix,1)
    image.close()
    
DSMrstr = rasterio.open(result_path+'ArcticDEM.tif')
predList = [Ground,groundMaskGMM8,ground_open8,groundMaskGMMloc]
predName = ['croppedgroundMaskGMM','croppedgroundMaskGMM8','croppedgroundMaskGMM8open','croppedgroundMaskGMMloc']

for i in np.arange(0,len(predList)):
    writeRaster(DSMrstr,np.float32(predList[i]),predName[i],result_path)