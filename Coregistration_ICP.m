clc,clear
path = 'D:/Workfolder_Zhang/Data/DigitalTerrainModel/test_region_alaska/study_region3_new/';

% package "topotoolbox" was adopted for reading and writing data, which can
% be downloaded from: https://topotoolbox.wordpress.com/download/
addpath(genpath('D:/Workfolder_Zhang/myCodes/codesOnline/topotoolbox-master'));
addpath(genpath(path));
addpath(genpath('D:/Workfolder_Zhang/myCodes/step_functions/attached_functions'));

SiteStr = 'SiteOne'; % SiteOne,
result_path = strcat(SiteStr,'_results/');

%% load data
DSM = GRIDobj(strcat(path,result_path,'ArcticDEM.tif'));
DTM = imread(strcat(path,result_path,'DTMlidar.tif'));

%% ground mask for registeration
G = dlmread(strcat(path,result_path,'ground_mask_for_coregistration.txt')); 
G = G == 1;
DSMm = DSM.Z;
DTMm = DTM;
[x,y] = getcoordinates(DSM,'matrix');

%% icp
rmseOrg = sqrt(immse(DSMm(G), DTMm(G)));
moving = pointCloud([x(G),y(G),DTMm(G)]);
fixed = pointCloud([x(G),y(G),DSMm(G)]);

[tform,movingReg,rmse] = pcregistericp(moving,fixed,'InlierRatio',0.6,'Metric','pointToPlane');

T2 = table(rmse,rmseOrg,'VariableNames',{'rmse_corrected','rmse_original'});
writetable(T2,strcat(path,result_path,'rmse_corrected_original_afterICP.txt'))

%% display the transformed results
figure(1),
pcshowpair(moving,fixed,'MarkerSize',10)
xlabel('X')
ylabel('Y')
zlabel('Z')
title('Point clouds before registration')
legend({'Moving point cloud','Fixed point cloud'},'TextColor','w')
legend('Location','southoutside')
print(strcat(path,result_path,'PointCloudsBeforeRegistration'),'-dpng',...
    '-r500');

figure(2),
pcshowpair(movingReg,fixed,'MarkerSize',10)
xlabel('X')
ylabel('Y')
zlabel('Z')
title('Point clouds after registration')
legend({'Moving point cloud','Fixed point cloud'},'TextColor','w')
legend('Location','southoutside')
print(strcat(path,result_path,'PointCloudsAfterRegistration'),'-dpng',...
    '-r500');

%%
movingDTM = pointCloud([x(:),y(:),DTMm(:)]);
fixedDSM = pointCloud([x(:),y(:),DSMm(:)]);

movingDTMreg = pctransform(movingDTM,tform);
movingDTMregLoc = double(movingDTMreg.Location);

xmin = x(1,1);
xmax = x(1,end);
ymin = y(end,1);
ymax = y(1,1);

vq_DTM = griddata(movingDTMregLoc(:,1),movingDTMregLoc(:,2),...
    movingDTMregLoc(:,3),x,y,'linear');

%% ctrl+/: comment multiple lines
corDTM = DSM;
corDTM.Z = vq_DTM;
GRIDobj2geotiff(corDSMlidar,strcat(path,result_path,'DSMlidar_icp.tif'));
