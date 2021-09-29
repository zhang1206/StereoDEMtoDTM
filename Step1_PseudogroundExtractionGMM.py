#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:53:49 2020

@author: zhangtianqi
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import rasterio
from skimage import io
import os
import os.path as osp
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

import sys
sys.path.append("D:/Workfolder_Zhang/myCodes/step_functions/attached_functions/DTMextraction/")
# import preProcessing

path = 'D:/Workfolder_Zhang/Data/DigitalTerrainModel/test_region_alaska/study_region3_new/' 
# use your path
os.chdir(path)

result_path = osp.join(path,'GMMresults/')
if osp.exists(result_path) == False:
    os.makedirs(result_path)   
    
#%% import data and calculate the input features
def spectraIdxCal(MB):
    red = MB[:,:,4]
    nir = MB[:,:,6]
    green = MB[:,:,2]
    NDVI = (nir - red)/(nir + red)
    MSAVI = (2*nir + 1 - np.sqrt(np.square(2*nir + 1) - 8*(nir - red)))/2
    NDWI = (green - nir)/(green + nir) 
    out = np.dstack((NDVI,MSAVI,NDWI))
    return out  

#SiteOne
MBmOne = io.imread(path+'SiteOne_results/'+'MB.tif') 
spectraIDXmOne = spectraIdxCal(MBmOne)

#SiteTwo
MBmTwo = io.imread(path+'SiteTwo_results/'+'MB.tif') 
spectraIDXmTwo = spectraIdxCal(MBmTwo)

#SiteThree
MBmThree = io.imread(path+'SiteThree_results/'+'MB.tif') 
spectraIDXmThree = spectraIdxCal(MBmThree)

#%% Feature preparation for GMM classification
imgOne = np.dstack((spectraIDXmOne,MBmOne))
imgTwo = np.dstack((spectraIDXmTwo,MBmTwo))
imgThree = np.dstack((spectraIDXmThree,MBmThree))

#%% MODEL SELECTION: find the optimal number of clusters
# from matplotlib.patches import Ellipse
from matplotlib import rcParams
rcParams['figure.figsize'] = 16, 8

# def draw_ellipse(position, covariance, ax=None, **kwargs):
#     """Draw an ellipse with a given position and covariance"""
#     ax = ax or plt.gca()
#     # Convert covariance to principal axes
#     if covariance.shape == (2, 2):
#         U, s, Vt = np.linalg.svd(covariance)
#         angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
#         width, height = 2 * np.sqrt(s)
#     else:
#         angle = 0
#         width, height = 2 * np.sqrt(covariance)
    
#     # Draw the Ellipse
#     for nsig in range(1, 4):
#         ax.add_patch(Ellipse(position, nsig * width, nsig * height,
#                              angle, **kwargs))
        
# def plot_gmm(gmm, X, label=True, ax=None):
#     ax = ax or plt.gca()
#     labels = gmm.fit(X).predict(X)
#     if label:
#         ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
#     else:
#         ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    
#     w_factor = 0.2 / gmm.weights_.max()
#     for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
#         draw_ellipse(pos, covar, alpha=w * w_factor)
#     plt.title("GMM with %d components"%len(gmm.means_), fontsize=(20))
#     plt.xlabel("U.A.")
#     plt.ylabel("U.A.")

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

img = np.hstack((imgOne,imgTwo,imgThree))
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

# plot the result
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

#%% save the clustering results
maxProbabimgOne = maxProbabimg[:,0:300]
maxProbabimgTwo = maxProbabimg[:,300:600]
maxProbabimgThree = maxProbabimg[:,600:900]

IEimgOne = IEimg[:,0:300]
IEimgTwo = IEimg[:,300:600]
IEimgThree = IEimg[:,600:900]

labelimgOne = labelimg[:,0:300]
labelimgTwo = labelimg[:,300:600]
labelimgThree = labelimg[:,600:900]


import matplotlib.colors as clr
from pylab import *

# maximum probability
predictors = [maxProbabimgOne,maxProbabimgTwo,maxProbabimgThree] 
title = ['Probability','Probability','Probability']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
fig.subplots_adjust(wspace=0.1,hspace=0.01)
for i,(ax, feature, title) in enumerate(zip(axes.flatten(), predictors, title)): 
    im = ax.imshow(feature,vmin=0.3,vmax=1)
    ax.set(title=title)
    ax.grid(False)
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])
  
fig.subplots_adjust(right=0.81)
cbar_ax = fig.add_axes([0.82, 0.2, 0.008, 0.6])
cbar = fig.colorbar(im, cax=cbar_ax)
plt.savefig(result_path+'Probability_threeSites.png',dpi = 300)

# clustering result
predictors = [labelimgOne,labelimgTwo,labelimgThree] 
title = ['Clusters','Clusters','Clusters']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
fig.subplots_adjust(wspace=0.1,hspace=0.01)
for i,(ax, feature, title) in enumerate(zip(axes.flatten(), predictors, title)): 
    im = ax.imshow(feature,cmap=cm.get_cmap('tab10',n_components))
    ax.set(title=title)
    ax.grid(False)
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])
  
fig.subplots_adjust(right=0.81)
cbar_ax = fig.add_axes([0.82, 0.2, 0.008, 0.6])
cbar = fig.colorbar(im, cax=cbar_ax,
                    boundaries=np.linspace(0.5,0.5+n_components,n_components+1),
             ticks=np.arange(n_components)+1)
plt.savefig(result_path+'Clusters_threeSites.png',dpi = 300)


# IE
predictors = [IEimgOne,IEimgTwo,IEimgThree] 
title = ['IE','IE','IE']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
fig.subplots_adjust(wspace=0.1,hspace=0.01)
for i,(ax, feature, title) in enumerate(zip(axes.flatten(), predictors, title)): 
    im = ax.imshow(feature,vmin=0,vmax=1.3)
    ax.set(title=title)
    ax.grid(False)
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])
  
fig.subplots_adjust(right=0.81)
cbar_ax = fig.add_axes([0.82, 0.2, 0.008, 0.6])
cbar = fig.colorbar(im, cax=cbar_ax)
plt.savefig(result_path+'IE_threeSites.png',dpi = 300)

#%% get the colormap in hex-format of the clustering map
cmap = cm.get_cmap('tab10',n_components)
for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    print(matplotlib.colors.rgb2hex(rgba))

#1f77b4
#d62728
#e377c2
#17becf

#%% Output result
def writeRaster(DSMrstr,image,file_str,result_path):
    # DSMrstr: rasterio object
    # interp: predictions, e.g., np.float32(interp), be sure to make the data type is consistent
    # file_str: e.g.,'IDWInterp_predGMM'
    # result_path: directory where you want to store the result
    output = rasterio.open(result_path+file_str+'.tif','w',driver='Gtiff',
                              width=image.shape[1], 
                              height = image.shape[0], 
                              count=1, crs=DSMrstr.crs, 
                              transform=DSMrstr.transform, 
                              dtype=DSMrstr.dtypes[0])
    output.write(image,1)
    output.close()

#-- SiteOne --- 
DSMrstr = rasterio.open(path+'SiteOne_results/'+'ArcticDEM.tif') 
List = [maxProbabimgOne,IEimgOne,labelimgOne]
Name = ['maxProbab','IE','cluster']
for i in np.arange(0,len(List)):
    writeRaster(DSMrstr,np.float32(List[i]),Name[i],path+'SiteOne_results/')
    
#-- SiteTwo --
DSMrstr = rasterio.open(path+'SiteTwo_results/'+'ArcticDEM.tif') 
List = [maxProbabimgTwo,IEimgTwo,labelimgTwo]
Name = ['maxProbab','IE','cluster']
for i in np.arange(0,len(List)):
    writeRaster(DSMrstr,np.float32(List[i]),Name[i],path+'SiteTwo_results/')
        
#-- SiteThree --
DSMrstr = rasterio.open(path+'SiteThree_results/'+'ArcticDEM.tif') 
List = [maxProbabimgThree,IEimgThree,labelimgThree]
Name = ['maxProbab','IE','cluster']
for i in np.arange(0,len(List)):
    writeRaster(DSMrstr,np.float32(List[i]),Name[i],path+'SiteThree_results/')

