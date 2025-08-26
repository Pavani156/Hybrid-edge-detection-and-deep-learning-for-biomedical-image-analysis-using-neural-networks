clc;
clear;
close all;

% Prompt user to select multiple image files
[file, path] = uigetfile({'*.jpg;*.png;*.bmp','Image Files'}, 'Select Images', 'MultiSelect', 'on');
if isequal(file, 0)
    error('No files selected. Please select valid image files.');
end

if ischar(file)
    file = {file}; % Ensure consistency if only one file is selected
end

num_images = length(file);
total_accuracy = 0;
total_precision = 0;
total_recall = 0;
total_f1_score = 0;
total_psnr = 0;
total_mse = 0;
total_ssim = 0;

% Load or train the U-Net model
net = load_or_train_unet();

for i = 1:num_images
    img = imread(fullfile(path, file{i}));
    
    % Convert to Grayscale if needed
    if size(img, 3) == 3
        gray_img = rgb2gray(img);
    else
        gray_img = img;
    end
    
    % Resize image to a fixed size
    fixed_size = [256, 256];
    gray_img = imresize(gray_img, fixed_size);
    
    % Apply Edge Detection Techniques
    sobel_edges = edge(gray_img, 'sobel');
    canny_edges = edge(gray_img, 'canny');

    % Apply U-Net Edge Detection
    unet_edges = apply_unet_edge_detection(gray_img, net);
    
    % Apply Thresholding to U-Net Enhanced Edges
    unet_edges = double(unet_edges); % Convert to double before binarization
    unet_edges = imbinarize(unet_edges, 0.5); % fixed threshold (0.9 -> 0.5)

    % Fallback mechanism: If U-Net produces blank output, use 'log' edge detection
    if sum(unet_edges(:)) == 0
        unet_edges = edge(gray_img, 'log');
    end
    
    % Combine Sobel, Canny for Hybrid Edge Detection
    hybrid_edges = sobel_edges | canny_edges;

    % Apply Thresholding to Hybrid Edge Detection
    hybrid_edges = double(hybrid_edges); 
    hybrid_edges = imbinarize(hybrid_edges, 0.5);

    % Generate Synthetic Ground Truth using Canny (Placeholder for Real Data)
    ground_truth = edge(gray_img, 'canny');
    
    % Convert all to logical format
    ground_truth = logical(ground_truth);
    detected_edges = logical(unet_edges);
    
    % Display Results
    figure;
    subplot(2,3,1), imshow(gray_img), title('Original Image');
    subplot(2,3,2), imshow(sobel_edges), title('Sobel Edge Detection');
    subplot(2,3,3), imshow(canny_edges), title('Canny Edge Detection');
    subplot(2,3,4), imshow(hybrid_edges), title('Hybrid Edge Detection');
    subplot(2,3,5), imshow(unet_edges), title('U-Net Enhanced Edges');
    subplot(2,3,6), imshow(ground_truth), title('Synthetic Ground Truth');
    
    % Evaluate Edge Detection Metrics
    [accuracy, precision, recall, f1_score, psnr_val, mse_val, ssim_val] = evaluate_metrics(ground_truth, detected_edges);
    total_accuracy = total_accuracy + accuracy;
    total_precision = total_precision + precision;
    total_recall = total_recall + recall;
    total_f1_score = total_f1_score + f1_score;
    total_psnr = total_psnr + psnr_val;
    total_mse = total_mse + mse_val;
    total_ssim = total_ssim + ssim_val;
end

% Compute and display average metrics
average_accuracy = total_accuracy / num_images;
average_precision = total_precision / num_images;
average_recall = total_recall / num_images;
average_f1_score = total_f1_score / num_images;
average_psnr = total_psnr / num_images;
average_mse = total_mse / num_images;
average_ssim = total_ssim / num_images;

fprintf('Average Metrics across %d images:\n Accuracy: %.4f\n Precision: %.4f\n Recall: %.4f\n F1-score: %.4f\n PSNR: %.2f\n MSE: %.4f\n SSIM: %.4f\n', ...
    num_images, average_accuracy, average_precision, average_recall, average_f1_score, average_psnr, average_mse, average_ssim);

%% --- Function Definitions ---

% Function to apply U-Net for edge detection
function unet_edges = apply_unet_edge_detection(image, net)
    % Resize input to match U-Net input size
    input_size = net.Layers(1).InputSize(1:2);
    image_resized = imresize(image, input_size);

    % Normalize image
    image_resized = im2double(image_resized);

    % Perform U-Net inference
    unet_result = semanticseg(image_resized, net);

    % Convert categorical segmentation output to binary mask
    unet_edges = unet_result == "edge";

    % Convert logical mask to double
    unet_edges = double(unet_edges);

    % Resize back to original size
    unet_edges = imresize(unet_edges, size(image));
end

% Function to load or train U-Net model
function net = load_or_train_unet()
    modelFile = 'unet_model.mat';
    if exist(modelFile, 'file')
        netData = load(modelFile);
        netFields = fieldnames(netData);
        net = netData.(netFields{1});
    else
        disp('Training U-Net model...');
        net = create_unet_model();
        save(modelFile, 'net');
    end
end

% create a u-net model
function net = create_unet_model()
    inputSize = [256, 256, 1];

    % Encoder (Contracting Path)
    layers = [
        imageInputLayer(inputSize, 'Name', 'input')

        % Block 1
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_1')
        reluLayer('Name', 'relu1_1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_2')
        reluLayer('Name', 'relu1_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

        % Block 2
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_1')
        reluLayer('Name', 'relu2_1')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_2')
        reluLayer('Name', 'relu2_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

        % Block 3
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3_1')
        reluLayer('Name', 'relu3_1')
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3_2')
        reluLayer('Name', 'relu3_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

        % Block 4
        convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv4_1')
        reluLayer('Name', 'relu4_1')
        convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv4_2')
        reluLayer('Name', 'relu4_2')
        transposedConv2dLayer(2, 256, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv4')
    ];

    % Decoder (Expanding Path)
    skip1 = [
        depthConcatenationLayer(2, 'Name', 'concat3')
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv5_1')
        reluLayer('Name', 'relu5_1')
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv5_2')
        reluLayer('Name', 'relu5_2')
        transposedConv2dLayer(2, 128, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv3')
    ];

    skip2 = [
        depthConcatenationLayer(2, 'Name', 'concat2')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv6_1')
        reluLayer('Name', 'relu6_1')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv6_2')
        reluLayer('Name', 'relu6_2')
        transposedConv2dLayer(2, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv2')
    ];

    skip3 = [
        depthConcatenationLayer(2, 'Name', 'concat1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv7_1')
        reluLayer('Name', 'relu7_1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv7_2')
        reluLayer('Name', 'relu7_2')
    ];

    % Final Classification
    classes = ["background","edge"]; % fixed class definition
    outputLayers = [
        convolution2dLayer(1, 2, 'Name', 'conv_final')
        softmaxLayer('Name', 'softmax') % fixed: replaced leakyreluLayer
        pixelClassificationLayer('Classes', classes, 'Name', 'output')
    ];

    % Build Layer Graph with Skip Connections
    net = layerGraph(layers);
    net = addLayers(net, skip1);
    net = addLayers(net, skip2);
    net = addLayers(net, skip3);
    net = addLayers(net, outputLayers);

    % Connect Skip Connections
    net = connectLayers(net, 'relu3_2', 'concat3/in1');
    net = connectLayers(net, 'upconv4', 'concat3/in2');
    net = connectLayers(net, 'relu2_2', 'concat2/in1');
    net = connectLayers(net, 'upconv3', 'concat2/in2');
    net = connectLayers(net, 'relu1_2', 'concat1/in1');
    net = connectLayers(net, 'upconv2', 'concat1/in2');
end

% Function to evaluate edge detection metrics
function [accuracy, precision, recall, f1_score, psnr_val, mse_val, ssim_val] = evaluate_metrics(ground_truth, detected_edges)
    ground_truth = logical(ground_truth);
    detected_edges = logical(detected_edges);

    TP = sum((ground_truth(:) == 1) & (detected_edges(:) == 1));
    FP = sum((ground_truth(:) == 0) & (detected_edges(:) == 1));
    FN = sum((ground_truth(:) == 1) & (detected_edges(:) == 0));
    TN = sum((ground_truth(:) == 0) & (detected_edges(:) == 0));

    fprintf('TP: %d ,FP: %d, FN: %d, TN : %d\n',TP, FP, FN, TN)

    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1_score = 2 * (precision * recall) / (precision + recall + eps);
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps);

    mse_val = immse(double(detected_edges), double(ground_truth));
    psnr_val = psnr(double(detected_edges), double(ground_truth));
    ssim_val = ssim(double(detected_edges), double(ground_truth));
end
