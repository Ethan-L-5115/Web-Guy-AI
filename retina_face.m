% % Import required Python packages
% py.importlib.import_module('tensorflow');
% py.importlib.import_module('numpy');
% facial_recognition = py.importlib.import_module('deepface');


% Load the pre-trained TensorFlow model
model_path = 'retina-face.h5';
model = importTensorFlowNetwork(model_path);

%%
% Set the path to your image folder
image_folder_path = 'dataset';
image_files = dir(fullfile(image_folder_path, '*.jpg')); % Assuming images are in .jpg format
num_images = length(image_files);

% Initialize a cell array to store preprocessed images
preprocessed_images = cell(num_images, 1);

% Read and preprocess all images in the folder
for i = 1:num_images
    image_path = fullfile(image_folder_path, image_files(i).name);
    img = imread(image_path);
    img_resized = imresize(img, [224, 224]); % Assuming the model input size is (224, 224, 3)
    img_array = py.numpy.array(img_resized, 'float32');
    
    % Store the preprocessed image in the cell array
    preprocessed_images{i} = img_array;
end

