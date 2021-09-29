clc,clear
path = 'D:/Workfolder_Zhang/Data/DigitalTerrainModel/test_region_alaska/study_region3_new/';
addpath(genpath('D:/Workfolder_Zhang/myCodes/codesOnline/topotoolbox-master'));
addpath(genpath(path));
addpath(genpath('D:/Workfolder_Zhang/myCodes/step_functions/attached_functions'));

SiteStr = 'SiteOne'; % SiteOne,
result_path = strcat(SiteStr,'_results/');

%% load data
DSM = GRIDobj(strcat(path,result_path,'ArcticDEM.tif'));
DTM = imread(strcat(path,result_path,'DTMlidar.tif'));
DSMlidar = imread(strcat(path,result_path,'VSMlidar.tif'));

%% ground mask for registeration
G = dlmread(strcat(path,result_path,'ground_mask_for_coregistration.txt')); 
G = G == 1;
DSMm = DSM.Z;
DTMm = DTM;
DSMlidarm = DSMlidar;
[x,y] = getcoordinates(DSM,'matrix');

%% icp
rmseOrg = sqrt(immse(DSMm(G), DTMm(G)));
moving = pointCloud([x(G),y(G),DTMm(G)]);
fixed = pointCloud([x(G),y(G),DSMm(G)]);

% %%
% inlierRatio = (0.2:0.05:0.7);
% rmse_corrected = zeros(length(inlierRatio),2);
% for i = 1:length(inlierRatio)
%     print(i)
%     [~,~,rmse] = pcregistericp(moving,fixed,'InlierRatio',inlierRatio(i),'Metric','pointToPlane');
%     error = round(rmse,4);
% %     print(error)
%     rmse_corrected(i,1) = round(rmse,4);
%     rmse_corrected(i,2) = inlierRatio(i);
% %     print(inlierRatio(i))
% end
%  
% [tform5,movingReg5,rmse5] = pcregistericp(moving,fixed,'InlierRatio',0.5);
% movingTform = pctransform(moving,tform); % test the transformation

%%
% InlierRatio = rmse_corrected(rmse_corrected == min(rmse_corrected(:,1)),2);
[tform,movingReg,rmse] = pcregistericp(moving,fixed,'InlierRatio',0.6,'Metric','pointToPlane');
% 
% T = table(rmse_corrected(:,1),rmse_corrected(:,2),'VariableNames',{'rmse_corrected','inlier_ratio'});
% writetable(T,strcat(path,result_path,'rmse_ground_coregistration_ICP.txt'))

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
movingDSMlidar = pointCloud([x(:),y(:),DSMlidarm(:)]);
fixedDSM = pointCloud([x(:),y(:),DSMm(:)]);

movingDTMreg = pctransform(movingDTM,tform);
movingDTMregLoc = double(movingDTMreg.Location);

movingDSMlidarreg = pctransform(movingDSMlidar,tform);
movingDSMlidarregLoc = double(movingDSMlidarreg.Location);

xmin = x(1,1);
xmax = x(1,end);
ymin = y(end,1);
ymax = y(1,1);

% Z = Dicp_DSMm; % flip vq is required sincee yq increases with increasing rows, while in northern
% hemisphere, yq should decrease with increasing rows

vq_DTM = griddata(movingDTMregLoc(:,1),movingDTMregLoc(:,2),...
    movingDTMregLoc(:,3),x,y,'linear');
vq_DSMlidar = griddata(movingDSMlidarregLoc(:,1),movingDSMlidarregLoc(:,2),...
    movingDSMlidarregLoc(:,3),x,y,'linear');

% DEMdiff = DSM;
% DEMdiff.Z = DSM.Z-vq_DTM;

% imagesc(DEMdiff),
% caxis([0,15])
% axis equal
% colorbar()

%% ctrl+/: comment multiple lines
corDTM = DSM;
corDTM.Z = vq_DTM;

corDSMlidar = DSM;
corDSMlidar.Z = vq_DSMlidar;

% set(0,'DefaultFigureColor',[1 1 1])
% imageschs(corDSMlidar)
% caxis([300,350])
% colorbar()

GRIDobj2geotiff(corDTM,strcat(path,result_path,'DTM_icp.tif'));
GRIDobj2geotiff(corDSMlidar,strcat(path,result_path,'DSMlidar_icp.tif'));
