function result = topoMasking_2(Dist,centerNum,mat)
%------ input parameters
% automatically generate a mask based on the dist to void center,
% Dist: distance threshold of other pixels to void center,
% centerNum: number of void center,

% e.g., Dist = 3; centerNum = 30 (sparse canopy);
% e.g., Dist = 10; centerNum = 5 (sparse canopy);

[r,c] = size(mat);
y = [1:r]'.*ones(size(mat));
x = ones(size(mat)).*[1:c];
x_vec = reshape(x,r*c,1);
y_vec = reshape(y,r*c,1);
coor_vec = [x_vec,y_vec];

%% randomly sample the centers
% s = RandStream('mlfg6331_64'); % for reproductivity
center_x_sample = datasample(x_vec,centerNum,'Replace',false);
center_y_sample = datasample(y_vec,centerNum,'Replace',false);
center_sample = [center_x_sample,center_y_sample];

dist = zeros(size(coor_vec,1),centerNum);
mask = zeros(size(coor_vec,1),centerNum);
for i = 1:centerNum
    center = center_sample(i,:);
    center_rep = repmat(center,size(coor_vec,1),1);
    dist(:,i) = sqrt(sum((coor_vec - center_rep).^2,2));
    ScalingFactor = randsample(100,1,false)*0.01;
    mask(:,i) = dist(:,i) <= ScalingFactor.*Dist;
end

%% define the distance
% mask = dist <= Dist;
mask_sum = sum(mask,2);
Mask = ~(mask_sum >= 1);
result = reshape(Mask,size(mat));

%% show the mask
imshow(result)