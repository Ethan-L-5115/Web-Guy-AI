% Define the directory where the images are located
img_dir = '/MATLAB Drive/CS495 Final/NCLS';
img_files = dir(fullfile(img_dir, '*.jpg'));

image_rgb_info = dir(fullfile(img_dir, '*.jpg'));
disp(img_dir)
disp(img_files)
% Define the parameters for the variations
%resize_factors = [0.5 0.75 1.2 1.4 1.7]; % Resize factors
rotate_angles = [3 6 9 12 15]; % Rotation angles
%translate_offsets = [50 100 150]; % Translation offsets
%to add resize and translate offsets add nested for looks above and below
%the roation angles
total_images = 0; % Initialize a variable to store the total number of processed images

% Create a loop to iterate through each image file
for i = 1:numel(img_files)
    % Load the image
    img = imread(fullfile(img_dir, img_files(i).name));
    
    % Create a loop to iterate through each parameter combination
        for k = 1:numel(rotate_angles)
                % Resize the image
                resized_img = imresize(img, resize_factors(j));
                
                % Rotate the image
                rotated_img = imrotate(resized_img, rotate_angles(k));
                
                % Translate the image
                translated_img = imtranslate(rotated_img, [translate_offsets(l) 0]);
                
                % Mirror the image
                mirrored_img = flip(translated_img, 2);
                
                % Save the processed images
                imwrite(mirrored_img, fullfile(img_dir, sprintf('processed_%d_%d_%d_mirrored.jpg', j, k, l)));
                imwrite(translated_img, fullfile(img_dir, sprintf(' processed_%d_%d_%d.jpg', j, k, l)));
                
                % Increment the total number of processed images
                total_images = total_images + 2;
        end
    
    fprintf('Processed %d of %d images\n', i, numel(img_files));
end

fprintf('Total processed images: %d\n', total_images); % Print the total number of processed images.
jpg_files = dir(fullfile(img_dir, '*.jpg'));

% Get a list of unique file names
unique_files = unique({jpg_files.name});

% Iterate over the unique file names and delete any duplicates
for i = 1:numel(unique_files)
    % Get the full path to the file
    full_file_path = fullfile(img_dir, unique_files{i});
    
    % Find all files with the same name in the directory
    matching_files = dir(fullfile(img_dir, unique_files{i}));

    
    % If there are more than one matching files, delete all but the first one
    if numel(matching_files) > 1
        for j = 2:numel(matching_files)
            delete(fullfile(img_dir, matching_files(j).name));
        end
    end
end
