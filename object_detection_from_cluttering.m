%%% Object Detection from a cluttered scene %%%
clc; clear all, close all;

%% Reading scene image
sourceImage = imread('source.jpg');
figure;
imshow(sourceImage);
title('Source Image (Target Object in Cluttered Scene)');

%% Reading object image
objectImage = imread('object1.jpg');
figure;
imshow(objectImage);
title('Object Image');

%% Converting to grayscale
sourceImage = rgb2gray(sourceImage);
objectImage = rgb2gray(objectImage);

%% Detecting feature points
sourcePoints = detectSURFFeatures(sourceImage);
objectPoints = detectSURFFeatures(objectImage);

%% Visualizing feature points of source image
figure;
imshow(sourceImage);
title('300 Strongest Feature Points from Source Image');
hold on;
plot(selectStrongest(sourcePoints, 300));

%% Visualizing feature points of object image
figure;
imshow(objectImage);
title('100 Strongest Feature Points from Object Image');
hold on;
plot(selectStrongest(objectPoints, 100));

%% Extracting feature descriptors
[sourceFeatures, sourcePoints] = extractFeatures(sourceImage, sourcePoints);
[objectFeatures, objectPoints] = extractFeatures(objectImage, objectPoints);

%% Matching feature descriptors
matchedFeatures = matchFeatures(objectFeatures, sourceFeatures);

%% Visualizing matched features
matchedobjectPoints = objectPoints(matchedFeatures(:, 1), :);
matchedsourcePoints = sourcePoints(matchedFeatures(:, 2), :);
figure;
showMatchedFeatures(objectImage, sourceImage, matchedobjectPoints, matchedsourcePoints, 'montage');
title('Matched Points (Including Outliers)');

%% Eliminating outliers and localizing object
[tform, inlierobjectPoints, inliersourcePoints] = estimateGeometricTransform(matchedobjectPoints, matchedsourcePoints, 'affine');

%% Visualizing final matched points
figure;
showMatchedFeatures(objectImage, sourceImage, inlierobjectPoints, inliersourcePoints, 'montage');
title('Matched Points (Inliers Only)');

%% Generating bounding polygon
% dimensions = [top-left; top-right; bottom-right; bottom-left; top-left again to close off the polygon]
objectPolygon = [1, 1; size(objectImage, 2), 1; size(objectImage, 2), size(objectImage, 1); 1, size(objectImage, 1); 1, 1];
    
%% Transforming polygon to local coordinates
newobjectPolygon = transformPointsForward(tform, objectPolygon);

%% Detecting localized object
figure;
imshow(sourceImage);
hold on;
line(newobjectPolygon(:, 1), newobjectPolygon(:, 2), 'Color', 'r', 'LineWidth', 1.75);
title('Detected Target Object');