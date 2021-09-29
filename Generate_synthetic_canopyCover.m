clc,clear,

% package "topotoolbox" was adopted for reading and writing data, which can
% be downloaded from: https://topotoolbox.wordpress.com/download/
addpath(genpath('D:/Workfolder_Zhang/myCodes/codeOnline/topotoolbox-master'));

mildTopo = GRIDobj('mildTopoUpdate.tif');
modTopo = GRIDobj('moderateTopoUpdate.tif');
steepTopo = GRIDobj('steepTopoUpdate.tif');

imagesc(mildTopo),colorbar,figure,
imagesc(modTopo),colorbar,figure,
imagesc(steepTopo),colorbar

%% generate the canopy masks
Num = 30; 
sparseCanopy = zeros(size(mildTopo.Z,1),size(mildTopo.Z,2),Num);
moderateCanopy = zeros(size(mildTopo.Z,1),size(mildTopo.Z,2),Num);
denseCanopy = zeros(size(mildTopo.Z,1),size(mildTopo.Z,2),Num);

sparseCanopyR = zeros(1,Num);
moderateCanopyR = zeros(1,Num);
denseCanopyR = zeros(1,Num);

center_num = 100;

temp1 = mildTopo;
temp2 = mildTopo;
temp3 = mildTopo;

area = size(mildTopo.Z,1)*size(mildTopo.Z,2);

for k = 1:Num
    sparseCanopy(:,:,k) = topoMasking_2(5,85,mildTopo.Z);
    temp1.Z = sparseCanopy(:,:,k);
    GRIDobj2geotiff(temp1,strcat(path,path1,'sparseCanopy_mask',num2str(k),'.tif'));
    sparseCanopyR(1,k) = sum(sum(1-sparseCanopy(:,:,k)))/area;
    
    moderateCanopy(:,:,k) = topoMasking_2(7.5,105,mildTopo.Z);
    temp2.Z = moderateCanopy(:,:,k);
    GRIDobj2geotiff(temp2,strcat(path,path2,'moderateCanopy_mask',num2str(k),'.tif'));
    moderateCanopyR(1,k) = sum(sum(1-moderateCanopy(:,:,k)))/area;
    
    denseCanopy(:,:,k) = topoMasking_2(13.5,150,mildTopo.Z);
    temp3.Z = denseCanopy(:,:,k);
    GRIDobj2geotiff(temp3,strcat(path,path3,'denseCanopy_mask',num2str(k),'.tif'));
    denseCanopyR(1,k) = sum(sum(1-denseCanopy(:,:,k)))/area;
end

%% save the canopy cover density
sparseCanopyR(Num+1) = mean(sparseCanopyR);
sparseCanopyR(Num+2) = std(sparseCanopyR);

moderateCanopyR(Num+1) = mean(moderateCanopyR);
moderateCanopyR(Num+2) = std(moderateCanopyR);

denseCanopyR(Num+1) = mean(denseCanopyR);
denseCanopyR(Num+2) = std(denseCanopyR);

dlmwrite(strcat(path,path1,'sparseCanopyRatio.txt'),sparseCanopyR);
dlmwrite(strcat(path,path2,'moderateCanopyRatio.txt'),moderateCanopyR);
dlmwrite(strcat(path,path3,'denseCanopyRatio.txt'),denseCanopyR);
