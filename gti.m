clc; 
clear; 
close all;

% Define input and output directories
inputFolder = 'C:\Users\PAVANI\OneDrive\Desktop\Team Project\Ultrasound\Classification\images';
outputFolder = 'C:\Users\PAVANI\OneDrive\Desktop\Team Project\ultra 1 gti\classification 1\images 1';  % Output folder for edge-detected images

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get list of all images in the input folder
imageFiles = dir(fullfile(inputFolder, '*.png')); % Change to '.jpg' if needed

% Check if images exist
if isempty(imageFiles)
    error('No images found in the input folder. Check the path and file format.');
end

% Process each image
for i = 1:length(imageFiles)
    % Read image
    img = imread(fullfile(inputFolder, imageFiles(i).name));

    % Convert to grayscale if needed
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end

    % Apply Canny edge detection
    edges = edge(img_gray, 'Canny');

    % Generate output filename and save
    outputFileName = fullfile(outputFolder, [imageFiles(i).name(1:end-4), '_edges.png']);
    imwrite(edges, outputFileName);

    fprintf('Processed: %s\n', imageFiles(i).name);
end

disp('Edge-detected images saved successfully!');