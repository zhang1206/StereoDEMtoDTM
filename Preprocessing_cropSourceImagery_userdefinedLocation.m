%% change path
clc,clear
path = 'D:/Workfolder_Zhang/Data/DigitalTerrainModel/test_region_alaska/study_region3_new/';
% addpath(genpath('D:\Workfolder_Zhang\myCodes\codesOnline\topotoolbox-master'));
addpath(genpath(path));
addpath(genpath('D:/Workfolder_Zhang/myCodes/step_functions/attached_functions'));

SiteStr = 'SiteThree'; % SiteOne, SiteThree
DSM = GRIDobj(strcat(path,SiteStr,'/croppedArcticDEM.tif'));
DSMm = DSM.Z;
startRow= 1;
endRow = 300; 
startCol = size(DSMm(6:(size(DSMm,1)-5),6:(size(DSMm,2)-5)),2)-319; 
endCol = size(DSMm(6:(size(DSMm,1)-5),6:(size(DSMm,2)-5)),2)-20; 

% SiteStr = 'SiteTwo'; 
% DSM = GRIDobj(strcat(path,SiteStr,'/croppedArcticDEM.tif'));
% DSMm = DSM.Z;
% startRow= 21;
% endRow = 320; 
% startCol = size(DSMm(6:(size(DSMm,1)-5),6:(size(DSMm,2)-5)),2)-301; 
% endCol = size(DSMm(6:(size(DSMm,1)-5),6:(size(DSMm,2)-5)),2)-2; 

result_path = strcat(SiteStr,'_results/');

R = worldfileread(strcat(path,result_path,'croppedDSM.tfw'), 'planar', size(DSM.Z));
[x,y] = getcoordinates(DSM,'matrix');

CoordRefSysCode = 32606;
targetSize = [300 300];

% ArcticDEM
J_DSM = DSMm(startRow:endRow,startCol:endCol);

% MB
MB = GRIDobj(strcat(path,SiteStr,'/croppedMB.tif'));
MBm = MB.Z;
J_MB = MBm(startRow:endRow,startCol:endCol,:);

% MP
MP = GRIDobj(strcat(path,SiteStr,'/croppedMP.tif'));
MPm = MP.Z;
J_MP = MPm(startRow:endRow,startCol:endCol);

% DTMlidar
DTMlidar = GRIDobj(strcat(path,SiteStr,'/croppedDTMlidar.tif'));
DTMlidarm = DTMlidar.Z;
J_DTMlidar = DTMlidarm(startRow:endRow,startCol:endCol);

% VSMlidar
VSMlidar = GRIDobj(strcat(path,SiteStr,'/croppedVSMlidar.tif'));
VSMlidarm = VSMlidar.Z;
J_VSMlidar = VSMlidarm(startRow:endRow,startCol:endCol);

xc = x(startRow:endRow,startCol:endCol);
yc = y(startRow:endRow,startCol:endCol);

xmin = xc(1,1);
xmax = xc(1,end);
ymin = yc(end,1);
ymax = yc(1,1);

R.XLimWorld = [xmin xmax];
R.YLimWorld = [ymin ymax];
R.RasterSize = targetSize;

% Write the data into geotiff 
geotiffwrite(strcat(path,result_path,'ArcticDEM.tif'),J_DSM,...
    R,'CoordRefSysCode',CoordRefSysCode)
geotiffwrite(strcat(path,result_path,'MB.tif'),J_MB,...
    R,'CoordRefSysCode',CoordRefSysCode)
geotiffwrite(strcat(path,result_path,'MP.tif'),J_MP,...
    R,'CoordRefSysCode',CoordRefSysCode)
geotiffwrite(strcat(path,result_path,'DTMlidar.tif'),J_DTMlidar,...
    R,'CoordRefSysCode',CoordRefSysCode)
geotiffwrite(strcat(path,result_path,'VSMlidar.tif'),J_VSMlidar,...
    R,'CoordRefSysCode',CoordRefSysCode)
