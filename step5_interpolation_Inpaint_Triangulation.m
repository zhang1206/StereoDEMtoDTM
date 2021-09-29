clc,clear
path = 'D:/Workfolder_Zhang/Data/DigitalTerrainModel/test_region_alaska/study_region3_new/';
addpath(genpath('D:/Workfolder_Zhang/myCodes/codeOnline/topotoolbox-master'));
addpath(genpath(path));
addpath(genpath('D:\Workfolder_Zhang\myCodes\step_functions\attached_functions'));

SiteStr = 'SiteThree';
result_path = strcat(SiteStr,'_results/');
result_path2 = strcat(SiteStr,'_results_PicTabs/');

addpath(genpath('D:\Workfolder_Zhang\myCodes\codesOnline\topotoolbox-master'));
DSM = GRIDobj(strcat(result_path,'ArcticDEM.tif'));

% fileStr = 'groundMaskGMM8open';
fileStr = 'groundMaskGMMloc';
groundMask = imread(strcat(path,result_path,'cropped',fileStr,'.tif'));
ground = groundMask == 1;

%% inpaint
Znan = DSM.Z;
nonground = groundMask == 0;
Znan(nonground) = nan;
DSMdetrd_inpaint = inpaint_nans(double(Znan),1); % del^2 PDE with least square estimates
DSMdetrd_Inpaint = DSM;
DSMdetrd_Inpaint.Z = DSMdetrd_inpaint;

GRIDobj2geotiff(DSMdetrd_Inpaint,strcat(path,result_path2,'inpaintInterp_',fileStr,'.tif'));

%--- with extrapolation
TINnaturalInterp = interp2Dgriddata_triangulationBased(DSM,ground,'natural','linear'); % TEMP stands for the detrended DSM 
TINlinrInterp = interp2Dgriddata_triangulationBased(DSM,ground,'linear','linear');
TINcubicInterp = interp2Dgriddata_triangulationBased(DSM,ground,'cubic','linear');

GRIDobj2geotiff(TINnaturalInterp,strcat(path,result_path2,'naturalInterp_',fileStr,'.tif'));
GRIDobj2geotiff(TINlinrInterp,strcat(path,result_path2,'linearInterp_',fileStr,'.tif'));
GRIDobj2geotiff(TINcubicInterp,strcat(path,result_path2,'cubicInterp_',fileStr,'.tif'));

%--- no extrapolation
naturalInterp = interp2Dgriddata_triangulationBased_noExtrapolate(DSM,ground,'natural'); % TEMP stands for the detrended DSM 
linrInterp = interp2Dgriddata_triangulationBased_noExtrapolate(DSM,ground,'linear');
cubicInterp = interp2Dgriddata_triangulationBased_noExtrapolate(DSM,ground,'cubic');

GRIDobj2geotiff(naturalInterp,strcat(path,result_path2,'naturalInterp_',fileStr,'_noExtrapolate.tif'));
GRIDobj2geotiff(linrInterp,strcat(path,result_path2,'linearInterp_',fileStr,'_noExtrapolate.tif'));
GRIDobj2geotiff(cubicInterp,strcat(path,result_path2,'cubicInterp_',fileStr,'_noExtrapolate.tif'));

%%
% naturalInterp = interp2Dgriddata(DSM,ground,'natural'); % TEMP stands for the detrended DSM 
% linrInterp = interp2Dgriddata(DSM,ground,'linear');
% cubicInterp = interp2Dgriddata(DSM,ground,'cubic');
% 
% GRIDobj2geotiff(naturalInterp,strcat(path,result_path,'naturalInterp_groundMaskGMM8open.tif'));
% GRIDobj2geotiff(linrInterp,strcat(path,result_path,'linearInterp_groundMaskGMM8open.tif'));
% GRIDobj2geotiff(cubicInterp,strcat(path,result_path,'cubicInterp_groundMaskGMM8open.tif'));

%% construct the coordinate matrix
% [xq,yq] = meshgrid(1:1:size(DSM,2), 1:1:size(DSM,1));

% extract the known locations, i.e., ground pixls
% 
% x = xq(ground);
% y = yq(ground);
% v = double(DSM(ground));
% vq = griddata(x,y,v,xq,yq,'cubic');

% mesh(xq,yq,vq)
% hold on
% plot3(x,y,v,'o')
% xlim([1 301])
% ylim([1 300])