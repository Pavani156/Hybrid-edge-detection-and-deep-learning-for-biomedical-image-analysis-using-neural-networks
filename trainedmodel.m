clc; clear; close all;

%  Define dataset folder paths
imageDir = 'C:\Users\PAVANI\OneDrive\Desktop\Team Project\Ultrasound\Classification\images';
labelDir = 'C:\Users\PAVANI\OneDrive\Desktop\Team Project\ultra 1 gti\classification 1\images 1';

%  Create Image Datastore
imds = imageDatastore(imageDir, 'FileExtensions', {'.png', '.jpg'});

%  Define Class Labels
classNames = ["background", "edge"];
pixelLabelID = [0, 255];

%  Create Pixel Label Datastore
pxds = pixelLabelDatastore(labelDir, classNames, pixelLabelID);

% 🛠 Verify dataset has images
numImages = numel(imds.Files);
numLabels = numel(pxds.Files);
disp(['Number of training images: ', num2str(numImages)]);
disp(['Number of training labels: ', num2str(numLabels)]);

% ⚠ Stop execution if dataset is empty
if numImages == 0 || numLabels == 0
    error('Dataset is empty! Check the dataset paths and ensure images exist.');
end

% 🛠 Reduce dataset size for testing (optional)
numSubset = min(numImages, 500);  % ✅ Use max 500 images if available
imds = subset(imds, numSubset);
pxds = subset(pxds, numSubset);

% 🛠 Resize Images
ds = pixelLabelImageDatastore(imds, pxds, 'OutputSize', [128 128]);

% ✅ Display training samples
disp(['Final number of training samples: ', num2str(length(imds.Files))]);

% 🧠 Define a Smaller U-Net Model
lgraph = unetLayers([128, 128, 1], 2);  % ✅ Smaller model for faster training

% ⚙ Define Optimized Training Options (Using CPU)
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'cpu', ...  % ✅ Use CPU instead of GPU
    'MaxEpochs', 10, ...  % ✅ Reduce epochs
    'MiniBatchSize', 16, ...  % ✅ Increase batch size
    'InitialLearnRate', 5e-3, ...  % ✅ Faster convergence
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% 🚀 Train U-Net Model
net = trainNetwork(ds, lgraph, options);

% 💾 Save Trained Model
save('C:\Users\PAVANI\OneDrive\Desktop\Team Project\unet_model.mat', 'net');

disp('✅ U-Net training complete!');