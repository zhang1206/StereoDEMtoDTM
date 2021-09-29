# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import os 
import os.path as osp
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# add the work directory
path = "D:/Workfolder_Zhang/Data/DigitalTerrainModel/test_region_alaska/study_region3_new/" 
os.chdir(path)

SiteStr = "SiteThree"

result_path = osp.join(path,SiteStr+'_results/')
if osp.exists(result_path) == False:
    os.makedirs(result_path)

result_path2 = osp.join(path,SiteStr+'_results_PicTabs/')
if osp.exists(result_path2) == False:
    os.makedirs(result_path2)
    
#%%
from sklearn.metrics import confusion_matrix
import pandas as pd

def EvaluationTable(y_true,y_pred,fileStr):    
    
    """
    e.g., fileStr = 'groundMaskGMM'   
    
    """
    # confusion matrix[[tn,fp],[fn,tp]]
    [tn, fp, fn, tp] = confusion_matrix(y_true, y_pred,normalize='all').flatten() 
    
    #overall accuracy: TP+TN/TP+FP+FN+TN
    OA = round((tp+tn),3)
    
    # producer accuracy or recall: TP/(TP+FN)
    PA = recall = round(tp/(tp+fn),3)
    
    # user accuracy or precision: TP/(TP+FP)
    UA = precision = round(tp/(tp+fp),3)
    
    # TI = 1-UA
    TIerror = 1-precision
    
    # TII = 1-PA
    TIIerror = 1 - recall
    
    # f1: 2*(Recall * Precision) / (Recall + Precision)
    f1 = round(2*(recall*precision)/(recall+precision),3)
    
    # TIerror = round(fp/(fp+tn),3)
    # TIIerror = round(fn/(tp+fn),3) # false negative rate
        
    TP = round(tp,3)
    data = [[fileStr,TP,TIerror,TIIerror,
             OA, UA, PA, f1]]
    
    groundMaskEvaluationTable = pd.DataFrame(data, columns = ['Method','TP','TI','TII',
                                                              'OA','UA','PA','F1'])
    # print('groundMaskEvaluationTable'+fileStr+':\n', groundMaskEvaluationTable)
    return groundMaskEvaluationTable

def comparison_all(prediction,truth): # both prediction and truth are vectors
    if len(prediction.shape)==1:
        flag = np.isnan(np.add(prediction,truth))
        RMSE = round(np.sqrt(np.mean((prediction[~flag]-truth[~flag])**2)),3)
        MAE = round(np.mean(abs(prediction[~flag]-truth[~flag])),3)
        
        bias = np.asarray(prediction)[~flag] - np.asarray(truth)[~flag]
        MBE = round(np.mean(bias),3)
        Range = np.nanmax(truth) - np.nanmin(truth)
        rRMSE = round(RMSE/Range,3)
    else:
        # prediction[prediction < 0] = np.nan
        # x = prediction.reshape(-1)
        # y = truth.reshape(-1)
        flag = np.isnan(np.add(prediction,truth))
        RMSE = round(np.sqrt(np.mean((prediction[~flag]-truth[~flag])**2)),3)
        MAE = round(np.mean(abs(prediction[~flag]-truth[~flag])),3)
        
        bias = np.asarray(prediction)[~flag] - np.asarray(truth)[~flag]
        MBE = round(np.mean(bias),3)
        Range = np.nanmax(truth) - np.nanmin(truth)
        rRMSE = round(RMSE/Range,3)
    return [RMSE,rRMSE,MAE,MBE]

from scipy.stats import pearsonr
def corrA(prediction, truth):
    if len(prediction.shape)==1:
        flag = np.isnan(np.add(prediction,truth))
        corr, _ = pearsonr(np.asarray(prediction)[~flag], np.asarray(truth)[~flag])
    else:
        x = prediction.reshape(-1)
        y = truth.reshape(-1)
        flag = np.isnan(np.add(x,y))
        corr, _ = pearsonr(np.asarray(x)[~flag], np.asarray(y)[~flag])
    corr = round(corr,3)
    return corr
def EvalTable(predList,predName,DTM):
    #predList = [DSM,predGMM,predGMM8,predGMM8open,predGMM8openlocRes]
    # predName = ['Orig','predGMM','predGMM8','predGMM8open','predGMM8openlocRes']
    data = np.zeros((len(predList),5))
    for i in np.arange(0,len(predList)):
        [RMSE,rRMSE,MAE,MBE] = comparison_all(predList[i],DTM)
        corr_all = corrA(predList[i],DTM)
        # R_sq = calRsquared(predList[i],DTM)
        data[i,:] = [RMSE,rRMSE,MAE,MBE,corr_all]   
    Table = pd.DataFrame(data, columns = ['RMSE(m)','rRMSE','MAE(m)','MBE(m)','R']) 
    Table.insert(0, 'Method', predName)
    return Table

#%%
from skimage import io

DSMrstr = rasterio.open(result_path+'ArcticDEM.tif')
MP = io.imread(result_path+'MP.tif')
DSM = io.imread(result_path+'ArcticDEM.tif')
DSMlidar = io.imread(result_path+ 'DSMlidar_icp.tif')
DTM = io.imread(result_path+ 'DTM_icp.tif')

#%% load the true ground mask and all other masks
CHMlidar = io.imread(result_path+'CHMlidar.tif')
groundMaskTrue = np.where(CHMlidar<=0.2,1,0)
plt.imshow(groundMaskTrue)

#%% load all masks 
groundMaskGMM = io.imread(result_path + 'croppedgroundMaskGMM.tif')
groundMaskGMM8 = io.imread(result_path + 'croppedgroundMaskGMM8.tif')
groundMaskGMM8open = io.imread(result_path + 'croppedgroundMaskGMM8open.tif')
groundMaskGMMloc = io.imread(result_path + 'croppedgroundMaskGMMloc.tif')
groundMaskCSFr1Stm = io.imread(result_path+'groundMaskCSFr1Stm.tif')
groundMaskMSD = io.imread(result_path+'groundMaskMSD.tif')
groundMaskMorpho = np.genfromtxt(result_path+'groundMaskMorpho.txt', delimiter=',')

#%% Evaluation table of all extracted masks
#--- ground location accuracy -----
flag = np.isnan(groundMaskTrue)
y_true = (groundMaskTrue)[~flag].astype(int)
y_pred_GMM = groundMaskGMM[~flag].astype(int)
y_pred_GMM8 = groundMaskGMM8[~flag].astype(int)
y_pred_GMM8open = groundMaskGMM8open[~flag].astype(int)
y_pred_GMMloc = groundMaskGMMloc[~flag].astype(int)
y_pred_CSFr1St = groundMaskCSFr1Stm[~flag].astype(int)
y_pred_MSD = groundMaskMSD[~flag].astype(int)
y_pred_Morpho = groundMaskMorpho[~flag].astype(int)

evalTabGMM = EvaluationTable(y_true,y_pred_GMM,'GMM')
evalTabGMM8 = EvaluationTable(y_true,y_pred_GMM8,'GMM8')
evalTabGMM8open = EvaluationTable(y_true,y_pred_GMM8open,'GMM8erosion')
evalTabGMMloc = EvaluationTable(y_true,y_pred_GMMloc,'GMMloc')
evalTabCSFr1St = EvaluationTable(y_true,y_pred_CSFr1St,'CSF')
evalTabMSD = EvaluationTable(y_true,y_pred_MSD,'MSD')
evalTabMorpho = EvaluationTable(y_true,y_pred_Morpho,'MBG')

evalTableAll = pd.concat([evalTabGMM,
                          evalTabGMM8,
                          evalTabGMM8open,
                          evalTabGMMloc,
                          evalTabCSFr1St,
                          evalTabMSD,
                          evalTabMorpho])
print('EvalTableAll:\n',evalTableAll)
evalTableAll.to_csv(result_path2 + 'groundMaskEvaluationAll_pixelMatchAccuracyMetrics.csv')

#%%--- DEM difference of the detected ground pixels -----
DSMGMM = np.where(groundMaskGMM ==0,np.nan,DSM)
DSMGMM8 = np.where(groundMaskGMM8 ==0,np.nan,DSM)
DSMGMM8erosion = np.where(groundMaskGMM8open==0,np.nan,DSM)
DSMGMMloc = np.where(groundMaskGMMloc==0,np.nan,DSM)
DSMCSF = np.where(groundMaskCSFr1Stm==0,np.nan,DSM)
DSMMSD = np.where(groundMaskMSD==0,np.nan,DSM)
DSMMBG = np.where(groundMaskMorpho==0,np.nan,DSM)

predListS = [DSMGMM,DSMGMM8,DSMGMM8erosion,DSMGMMloc,DSMCSF,DSMMSD,DSMMBG]
predNameS = ['GMM','GMM8',"GMM8erosion",'GMMloc','CSF','MSD','MBG']

df_withSmooth_all = EvalTable(predListS, predNameS, DTM)
print('DEMdiff: \n',df_withSmooth_all)
df_withSmooth_all.to_csv(result_path2 + 'groundMaskEvaluationAll_DEMdiffAccuracyMetrics.csv')

#%% plot the DEM difference over identified ground masks
groundmaskTrueDEMdiff = np.where(groundMaskTrue==0,np.nan,CHMlidar)
groundMask1DEMdiff = np.where(groundMaskGMMloc==0,np.nan,DSM-DTM)
groundMask2DEMdiff = np.where(groundMaskCSFr1Stm==0,np.nan,DSM-DTM)
groundMask4DEMdiff = np.where(groundMaskMSD==0,np.nan,DSM-DTM)
groundMask5DEMdiff = np.where(groundMaskMorpho==0,np.nan,DSM-DTM)

predictors = [groundmaskTrueDEMdiff,groundMask1DEMdiff,
              groundMask2DEMdiff,groundMask4DEMdiff,groundMask5DEMdiff]

title = ['LiDAR-derived','GMM','CSF','MSD','MBG']
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(4,15))
fig.subplots_adjust(top=0.985,
# bottom=0.1,
left=0.02,
right=0.963,
hspace=0.05,
wspace=0.08)
for ax, feature, title in zip(axes.flatten(), predictors, title):
    cmap =  matplotlib.cm.viridis
    cmap.set_over('r')
    cmap.set_under('b')
    im = ax.imshow(feature,vmin=-1, vmax=3,cmap = cmap)
    # ax.set(title=title)
    ax.grid(False)
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])
    
# fig.subplots_adjust(bottom=0.08)
# cbar_ax = fig.add_axes([0.1, 0.06, 0.8, 0.01]) # left, bottom, width, height
# fig.colorbar(im, cax = cbar_ax,
#               ticks=[-1,0,1,2,3],
#               extend= 'both',
#               orientation='horizontal')

plt.tight_layout()
plt.savefig(result_path2 +'DEMdiff_groundMaskAl_test1.png',dpi = 500)

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# divider = make_axes_locatable(ax)
# cax = divider.new_vertical(size="5%",pad=0.02, pack_start=True)
# fig.add_axes(cax)
# fig.colorbar(im, cax=cax, ticks=[-1,0,1,2,3],
#               extend= 'both', orientation="horizontal")

#--- plot the DEM difference over identified ground masks ----
groundMaskGMMDEMdiff = np.where(groundMaskGMM==0,np.nan,DSM-DTM)
groundMaskGMM8DEMdiff = np.where(groundMaskGMM8==0,np.nan,DSM-DTM)

predictors = [groundMaskGMMDEMdiff,groundMaskGMM8DEMdiff]

title = ['GMM','GMM + Uncertainty']
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4,5))
fig.subplots_adjust(top=0.985,
bottom=0.1,
left=0.02,
right=0.963,
hspace=0.05,
wspace=0.08)
for ax, feature, title in zip(axes.flatten(), predictors, title):
    cmap =  matplotlib.cm.viridis
    cmap.set_under('b')
    cmap.set_over('r')
    im = ax.imshow(feature,vmin=-1, vmax=3,cmap = cmap)
    # ax.set(title=title)
    ax.grid(False)
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])
    
fig.subplots_adjust(bottom=0.1)
# cbar_ax = fig.add_axes([0.82, 0.41, 0.02, 0.18]) # left, bottom, width, height
cbar_ax = fig.add_axes([0.14, 0.05, 0.73, 0.015]) # left, bottom, width, height
fig.colorbar(im, cax = cbar_ax,
              ticks=[-1,0,1,2,3],
              extend= 'both',
              orientation='horizontal')

# plt.tight_layout()
plt.savefig(result_path2+'DEMdiff_groundMaskGMM_test1.png',dpi = 500)
