%%% Object Detection from a cluttered scene %%%
clc; clear all, close all;

%% Reading scene image
sourceImage = imread('source.jpg');
figure;
imshow(sourceImage);
title('Source Image (Target Object in Cluttered Scene)');

%% Reading object image
objectImage1 = imread('object1.jpg');
objectImage2 = imread('object2.jpg');
figure, imshow(objectImage1), title('Object Image 1');
figure, imshow(objectImage2), title('Object Image 2');

%% Converting to grayscale
sourceImage = rgb2gray(sourceImage);
objectImage1 = rgb2gray(objectImage1);
objectImage2 = rgb2gray(objectImage2);

%% Detecting feature points
sourcePoints = detectSURFFeatures(sourceImage);
objectPoints1 = detectSURFFeatures(objectImage1);
objectPoints2 = detectSURFFeatures(objectImage2);

%% Visualizing feature points of source image
figure;
imshow(sourceImage);
title('300 Strongest Feature Points from Source Image');
hold on;
plot(selectStrongest(sourcePoints, 300));

%% Visualizing feature points of object image 1
figure;
imshow(objectImage1);
title('100 Strongest Feature Points from Object Image 1');
hold on;
plot(selectStrongest(objectPoints1, 100));

%% Visualizing feature points of object image 2
figure;
imshow(objectImage2);
title('100 Strongest Feature Points from Object Image 2');
hold on;
plot(selectStrongest(objectPoints2, 100));

%% Extracting feature descriptors
[sourceFeatures, sourcePoints] = extractFeatures(sourceImage, sourcePoints);
[objectFeatures1, objectPoints1] = extractFeatures(objectImage1, objectPoints1);
[objectFeatures2, objectPoints2] = extractFeatures(objectImage2, objectPoints2);

%% Matching feature descriptors
matchedFeatures1 = matchFeatures(objectFeatures1, sourceFeatures);
matchedFeatures2 = matchFeatures(objectFeatures2, sourceFeatures, 'MaxRatio', 0.9);

%% Visualizing matched features
matchedobjectPoints1 = objectPoints1(matchedFeatures1(:, 1), :);
matchedsourcePoints1 = sourcePoints(matchedFeatures1(:, 2), :);
matchedobjectPoints2 = objectPoints2(matchedFeatures2(:, 1), :);
matchedsourcePoints2 = sourcePoints(matchedFeatures2(:, 2), :);
figure, showMatchedFeatures(objectImage1, sourceImage, matchedobjectPoints1, matchedsourcePoints1, 'montage'), title('Matched Points 1 (Including Outliers)');
figure, showMatchedFeatures(objectImage2, sourceImage, matchedobjectPoints2, matchedsourcePoints2, 'montage'), title('Matched Points 2 (Including Outliers)');

%% Eliminating outliers and localizing object
[tform1, inlierobjectPoints1, inliersourcePoints1] = estimateGeometricTransform(matchedobjectPoints1, matchedsourcePoints1, 'affine');
[tform2, inlierobjectPoints2, inliersourcePoints2] = estimateGeometricTransform(matchedobjectPoints2, matchedsourcePoints2, 'similarity');

%% Visualizing final matched points
figure, showMatchedFeatures(objectImage1, sourceImage, inlierobjectPoints1, inliersourcePoints1, 'montage'), title('Matched Points 1 (Inliers Only)');
figure, showMatchedFeatures(objectImage2, sourceImage, inlierobjectPoints2, inliersourcePoints2, 'montage'), title('Matched Points 2 (Inliers Only)');

%% Generating bounding polygon
% dimensions = [top-left; top-right; bottom-right; bottom-left; top-left again to close off the polygon]
objectPolygon1 = [1, 1; size(objectImage1, 2), 1; size(objectImage1, 2), size(objectImage1, 1); 1, size(objectImage1, 1); 1, 1];
objectPolygon2 = [1, 1; size(objectImage2, 2), 1; size(objectImage2, 2), size(objectImage2, 1); 1, size(objectImage2, 1); 1, 1];

%% Transforming polygon to local coordinates
newobjectPolygon1 = transformPointsForward(tform1, objectPolygon1);
newobjectPolygon2 = transformPointsForward(tform2, objectPolygon2);

%% Detecting localized objects
figure;
imshow(sourceImage);
hold on;
line(newobjectPolygon1(:, 1), newobjectPolygon1(:, 2), 'Color', 'g', 'LineWidth', 1.75);
line(newobjectPolygon2(:, 1), newobjectPolygon2(:, 2), 'Color', 'y', 'LineWidth', 1.75);
title('Detected Target Objects');