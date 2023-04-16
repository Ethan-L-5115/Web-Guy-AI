% Load the dataset
imageFolderPath = 'dataset';
imageFiles = dir(fullfile(imageFolderPath, '*.jpg'));

%% Display a random subset of images
numImages = length(imageFiles);
numSamples = 25;
randIndices = randperm(numImages, numSamples);
figure;

for i = 1:numSamples
    imgPath = fullfile(imageFolderPath, imageFiles(randIndices(i)).name);
    img = imread(imgPath);
    subplot(5, 5, i);
    imshow(img);
    title(sprintf('Image %d', randIndices(i)));
end

%% Determine the distribution of image sizes
imgWidths = zeros(numImages, 1);
imgHeights = zeros(numImages, 1);

for i = 1:numImages
    imgPath = fullfile(imageFolderPath, imageFiles(i).name);
    img = imread(imgPath);
    [imgHeights(i), imgWidths(i), ~] = size(img);
end

figure;
hist3([imgHeights, imgWidths], 'CDataMode', 'auto', 'FaceColor', 'interp');
xlabel('Image Heights');
ylabel('Image Widths');
zlabel('Frequency');
title('Distribution of Image Sizes');

%% Compute and display the partitioned LST buckets
rgbBuckets = partition_by_rgb_buckets(imageFiles, imageFolderPath);

rgbBuckets
%%
plot_heatmaps(rgbBuckets)

%% Download and load a pre-trained deep learning model (e.g., ResNet-50)
net = resnet50();

% Classify activities in the dataset
activityLabels = classify_activities(imageFiles, imageFolderPath, net);

% Visualize the distribution of activities
figure;
histogram(categorical(activityLabels));
xlabel('Activity');
ylabel('Frequency');
title('Activity Distribution');

%%


% Function to classify activities in images
function activityLabels = classify_activities(imageFiles, imageFolderPath, net)
    numImages = length(imageFiles);
    activityLabels = cell(numImages, 1);

    for i = 1:numImages
        imgPath = fullfile(imageFolderPath, imageFiles(i).name);
        img = imread(imgPath);

        % Resize the image to match the input size of the network
        imgResized = imresize(img, net.Layers(1).InputSize(1:2));
        
        % Classify the image using the pre-trained network
        label = classify(net, imgResized);
        activityLabels{i} = char(label);
    end
end

% Function to compute average LST values for each image and categorize them into buckets
function [buckets] = partition_by_rgb_buckets(imageFiles, imageFolderPath)
    numImages = length(imageFiles);

    % Initialize the buckets
    buckets = cell(3, 3, 3);

    for i = 1:numImages
        imgPath = fullfile(imageFolderPath, imageFiles(i).name);
        img = imread(imgPath);
        
        % Compute average LST values for the image
        avgR = mean2(img(:,:,1));
        avgG = mean2(img(:,:,2));
        avgB = mean2(img(:,:,3));

        % Determine the bucket for the image
        rBucket = min(floor(avgR / 85) + 1, 3);
        gBucket = min(floor(avgG / 85) + 1, 3);
        bBucket = min(floor(avgB / 85) + 1, 3);

        % Add the image to the corresponding bucket
        buckets{rBucket, gBucket, bBucket} = [buckets{rBucket, gBucket, bBucket}; imageFiles(i)];
    end
end

function plot_heatmaps(buckets)
    % Initialize the matrices for L, S, and T dimensions
    lMatrix = zeros(3, 3);
    sMatrix = zeros(3, 3);
    tMatrix = zeros(3, 3);
    
    % Calculate the number of images in each L, S, and T bucket
    for l = 1:3
        for s = 1:3
            for t = 1:3
                numImages = length(buckets{l, s, t});
                lMatrix(l, s) = lMatrix(l, s) + numImages;
                sMatrix(l, t) = sMatrix(l, t) + numImages;
                tMatrix(s, t) = tMatrix(s, t) + numImages;
            end
        end
    end
    
    % Plot the 2D heatmaps for L, S, and T dimensions
    figure;
    
    subplot(1, 3, 1);
    imagesc(lMatrix);
    colorbar;
    xlabel('Green');
    ylabel('Red');
    title('Red vs. Green');
    
    subplot(1, 3, 2);
    imagesc(sMatrix);
    colorbar;
    xlabel('Blue');
    ylabel('Red');
    title('Red vs. Blue');
    
    subplot(1, 3, 3);
    imagesc(tMatrix);
    colorbar;
    xlabel('Blue');
    ylabel('Green');
    title('Green vs. Blue');
end


% 
% 
% % Example function to visualize images based on a given property
% function visualize_images_by_property(imageFiles, imageFolderPath, propertyFilter)
%     filteredImages = propertyFilter(imageFiles, imageFolderPath);
% 
%     numFilteredImages = length(filteredImages);
%     gridSize = ceil(sqrt(numFilteredImages));
%     figure;
% 
%     for i = 1:numFilteredImages
%         imgPath = fullfile(imageFolderPath, filteredImages(i).name);
%         img = imread(imgPath);
%         subplot(gridSize, gridSize, i);
%         imshow(img);
%         title(sprintf('Image %d', i));
%     end
% end