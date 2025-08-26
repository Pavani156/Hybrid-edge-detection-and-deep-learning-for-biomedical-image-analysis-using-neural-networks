% Prompt the user to select an image file
[filename, pathname] = uigetfile({'*.jpg;*.png;*.jpeg;*.bmp', 'Image Files (*.jpg, *.png, *.jpeg, *.bmp)'}, ...
                                  'Select an Image');
if isequal(filename,0)
   disp('User selected Cancel');
   return;
end

% Read the selected image
img = imread(fullfile(pathname, filename));

% Convert to grayscale if the image is RGB
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Create a figure with extended size
figure;
set(gcf, 'Position', [100, 100, 1200, 800]); % Adjust figure size (width, height)

% Display the original image
subplot(3,3,1);
imshow(img);
title('Original Image');

% Sobel Edge Detection
sobel_edges = edge(img, 'Sobel');
subplot(3,3,2);
imshow(sobel_edges);
title('Sobel Edge Detection');

% Prewitt Edge Detection
prewitt_edges = edge(img, 'Prewitt');
subplot(3,3,3);
imshow(prewitt_edges);
title('Prewitt Edge Detection');

% Roberts Edge Detection
roberts_edges = edge(img, 'Roberts');
subplot(3,3,4);
imshow(roberts_edges);
title('Roberts Edge Detection');

% Canny Edge Detection (show in the same figure)
canny_edges = edge(img, 'Canny');
subplot(3,3,5);
imshow(canny_edges);
title('Canny Edge Detection');
