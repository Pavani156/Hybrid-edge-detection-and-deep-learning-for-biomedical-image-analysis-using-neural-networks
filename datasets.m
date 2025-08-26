clc; clear; close all;

% Define dataset paths
imageDir = 'C:\Users\PAVANI\OneDrive\Desktop\Team Project\Ultrasound\Classification\images';
labelDir = 'C:\Users\PAVANI\OneDrive\Desktop\Team Project\ultra 1 gti\classification 1\images 1';

% Create image and label datastores
imds = imageDatastore(imageDir, 'FileExtensions', {'.png', '.jpg'});
pxds = imageDatastore(labelDir, 'FileExtensions', {'.png', '.jpg'});

% Check if the dataset is loaded properly
disp(['Number of images: ', num2str(length(imds.Files))]);
disp(['Number of labels: ', num2str(length(pxds.Files))]);