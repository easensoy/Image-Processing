format longg;
format compact;
clear all;

fontSize = 20;
load sampleFrames.mat
%img = imread('lena_gray.bmp');
%img = imread('sunflower2.png');
 img = vidFrame1(:,:,2);
img_threshold = img;
img_threshold(img_threshold(:)>100) = 100;

subplot(3,3,1);
imshow(img,'InitialMagnification','fit')
title('Original image');
subplot(3,3,2);
imshow(img_threshold,'InitialMagnification','fit')
title('Filtered image by threshold >100');
%%  Edge detection
% using MATLAB toolbox
img_edge = edge(img, 'Sobel');
subplot(3,3,3);
imshow(img_edge,'InitialMagnification','fit')
title('Edges of image - MATLAB Toolbox');

% from scratch
g_y = [-1 -2 -1;0 0 0;1 2 1];
g_x = [-1 0 1;-2 0 2;-1 0 1];
img_conv = [];
for i = 2:size(img,1)-1
    for j = 2:size(img,2)-1
        myImg = img(i-1:i+1,j-1:j+1);
        img_conv_x = sum(sum(double(myImg).*g_x));          % the convolution at each point equals to the peak of the convolution
        img_conv_y = sum(sum(double(myImg).*g_y));           % that is when the centre of the kernel is located at the centre of the myImg
        img_conv(i,j) = (sqrt(img_conv_x^2+img_conv_y^2));
    end
end
img_conv(img_conv(:)<=0.2*max(max(img_conv))) = 0;
subplot(3,3,4);
imshow(img_conv,'InitialMagnification','fit')
title('Edges of image - In-Home implementation');

%% Blob detection
% Approximation
LoG_approx = [0 1 1 2 2 2 1 1 0;
              1 2 4 5 5 5 4 2 1;
              1 4 5 3 0 3 5 4 1;
              2 5 3 -12 -24 -12 3 5 2;
              2 5 0 -24 -40 -24 0 5 2;
              2 5 3 -12 -24 -12 3 5 2;
              1 4 5 3 0 3 5 4 1;
              1 2 4 5 5 5 4 2 1;
              0 1 1 2 2 2 1 1 0];
blob_approx = [];
for i = round(length(LoG_approx)/2):size(img,1)-(round(length(LoG_approx)/2)-1)
    for j = round(length(LoG_approx)/2):size(img,2)-(round(length(LoG_approx)/2)-1)
        myImg = img(i-(round(length(LoG_approx)/2)-1):i+(round(length(LoG_approx)/2)-1),j-(round(length(LoG_approx)/2)-1):j+(round(length(LoG_approx)/2)-1));
        blob_approx(i,j) = sum(sum(double(myImg).*LoG_approx));
    end
end
% blob_approx(blob_approx(:)<-50 & blob_approx(:)>10) = 127;
blob_approx(blob_approx(:)>=0.01*max(max(blob_approx))) = 0;
blob_approx(blob_approx(:)<=0.01*min(min(blob_approx))) = 255;
CC = bwconncomp(blob_approx);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
blob_approx(CC.PixelIdxList{idx}) = 0;
subplot(3,3,5);
imshow(blob_approx,'InitialMagnification','fit')
title('Blob detection - In-Home approximation');

stats = regionprops('table',blob_approx,'Centroid',...
    'MajorAxisLength','MinorAxisLength');
centers = stats.Centroid;
diameters = stats.MinorAxisLength;
radii = diameters/2;
% hold on
% viscircles(centers,radii);

% from scratch
gaussian = [];
sigma = 2.5;
gaussinaKernelSize = 31;
for i = 1:gaussinaKernelSize
    for j = 1:gaussinaKernelSize
        gaussian(i,j) = gaussinaKernelSize*(1/2/pi/sigma)*exp(-(((i-gaussinaKernelSize/2))^2+((j-gaussinaKernelSize/2))^2)/2/sigma^2);
    end
end
gaussian(gaussian(:)<1)=0;
LoG = sigma^2.*del2(gaussian);
if mod(gaussinaKernelSize,2) == 0
    LoG = LoG(1:gaussinaKernelSize-1,1:gaussinaKernelSize-1);
end
% img_conv = round(conv2(gaussian, img, 'same'));
blob = [];
for i = round(gaussinaKernelSize/2):size(img,1)-(round(gaussinaKernelSize/2)-1)
    for j = round(gaussinaKernelSize/2):size(img,2)-(round(gaussinaKernelSize/2)-1)
        myImg = img(i-(round(gaussinaKernelSize/2)-1):i+(round(gaussinaKernelSize/2)-1),j-(round(gaussinaKernelSize/2)-1):j+(round(gaussinaKernelSize/2)-1));
        blob(i,j) = sum(sum(double(myImg).*LoG));
    end
end
% blob(blob(:)>=-30 & blob(:)<=30) = 127;
blob(blob(:)>=0.025*max(max(blob))) = 0;
blob(blob(:)<0.025*min(min(blob))) = 255;
CC = bwconncomp(blob,4);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
blob(CC.PixelIdxList{idx}) = 0;
subplot(3,3,6);
imshow(blob,'InitialMagnification','fit')
title('Blob detection - In-Home LoG');

stats = regionprops('table',blob,'Centroid',...
    'MajorAxisLength','MinorAxisLength', 'ConvexHull', 'ConvexImage', 'Area','BoundingBox');
centers = stats.Centroid;
diameters = stats.MinorAxisLength;
radii = diameters/2;
% hold on
% viscircles(centers,radii);
% hold on
% imshow(stats.ConvexImage{end})
% hold on
% rectangle('Position',stats.BoundingBox(end,:),'EdgeColor','b',...
%     'LineWidth',3)
% figure
% mesh(blob(1:10:end,1:10:end))

%% Blob Detection using OpenCV
keypoints = blobDetectionOCV(img);
subplot(3,3,7);
imshow(img,'InitialMagnification','fit')
hold on
scatter(keypoints.Location(:,1),keypoints.Location(:,2),10.*keypoints.Scale(:),'filled')
title('Blob detection - OpenCV in Matlab');

% keypoints1 = detectORBFeaturesOCV(img);
% hold on
% scatter(keypoints1.Location(:,1),keypoints1.Location(:,2),keypoints1.Scale(:),'filled')

%% Blob Detection using MATLAB Vision Toolbox
hBlobAnalysis = vision.BlobAnalysis('MinimumBlobArea',100,...
    'Connectivity', 4,'MaximumCount', 50, 'ExcludeBorderBlobs', false, 'MaximumBlobArea', 5000);
% [objArea,objCentroid,bboxOut] = step(hBlobAnalysis,img);
[objArea,objCentroid,bboxOut] = hBlobAnalysis(logical(blob));
subplot(3,3,8);
imshow(img,'InitialMagnification','fit')
hold on
scatter(objCentroid(:,1),objCentroid(:,2),objArea(:),'filled')
title('Blob detection - MATLAB Vision Toolbox');

release(hBlobAnalysis)
