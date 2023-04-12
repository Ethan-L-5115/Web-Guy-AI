% Load the dataset
imageFolderPath = 'dataset';
imageFiles = dir(fullfile(imageFolderPath, '*.jpg'));
%%
features = extract_image_features(imageFiles, imageFolderPath);
save('features.mat', 'features');
1
%%
features = load('features.mat').features;

%%
% Perform clustering with K-means
desired_clusters = 15;
[exemplars, assignments] = cluster_with_kmeans(features, desired_clusters);

% Display the exemplars and assigned images
display_exemplars_and_assigned_images(imageFiles, imageFolderPath, exemplars, assignments);
%%

% Function to extract features using a pre-trained VGG-16 model
function features = extract_image_features(imageFiles, imageFolderPath)
    % Load the pre-trained VGG-16 model
    vgg16_net = vgg16();
    % analyzeNetwork(vgg16_net);

    % Remove the last three layers to extract features instead of performing classification
    feature_layer = 'fc7';
    vgg16_net_layers = vgg16_net.Layers;%(1:end-5);

    % Add a ReLU layer as the output layer
    output_layer = reluLayer('Name', 'output');
    modified_layers = [vgg16_net_layers];

    % Create a new network with the modified layers
    modified_net = assembleNetwork(modified_layers);

    % Initialize feature matrix
    num_images = length(imageFiles);
    num_features = 4096; % The output size of 'fc7' layer in VGG-16
    features = zeros(num_images, num_features);

    for i = 1:num_images
        img_path = fullfile(imageFolderPath, imageFiles(i).name);
        img = imread(img_path);

        % Resize the image to the input size of VGG-16
        img_resized = imresize(img, [224, 224]);

        % Extract features using the modified VGG-16 model
        img_features = activations(modified_net, img_resized, feature_layer, 'OutputAs', 'rows');

        % Store the features in the feature matrix
        features(i, :) = img_features;
    end
end

function [exemplars, assignments] = cluster_with_kmeans(features, k)
    % Perform K-means clustering
    [assignments, centroids] = kmeans(features, k);
    
    % Find exemplars (closest data point to each centroid)
    exemplars = zeros(1, k);
    for i = 1:k
        distances = pdist2(features, centroids(i, :), 'euclidean');
        [~, exemplar_idx] = min(distances);
        exemplars(i) = exemplar_idx;
    end
end

function display_exemplars_and_assigned_images(imageFiles, imageFolderPath, exemplars, assignments)
    num_clusters = length(exemplars);

    for i = 1:num_clusters
        exemplar_idx = exemplars(i);
        cluster_indices = find(assignments == i);

        % Remove the exemplar index from the cluster indices
        cluster_indices(cluster_indices == exemplar_idx) = [];

        % Select 2 random images from the same cluster
        num_random_images = 2;
        random_indices = datasample(cluster_indices, num_random_images, 'Replace', false);

        % Display the exemplar and 2 random images from the same cluster
        figure(i);
        sgtitle(['Cluster ' num2str(i)]);
        
        % Show the exemplar
        exemplar_path = fullfile(imageFolderPath, imageFiles(exemplar_idx).name);
        exemplar_img = imread(exemplar_path);
        subplot(1, 3, 1);
        imshow(exemplar_img);
        title('Exemplar');
        
        % Show the 2 random images
        for j = 1:num_random_images
            random_img_path = fullfile(imageFolderPath, imageFiles(random_indices(j)).name);
            random_img = imread(random_img_path);
            subplot(1, 3, j+1);
            imshow(random_img);
            title(['Random Image ' num2str(j)]);
        end
    end
end
% 
% 
% function display_exemplars_and_assigned_images(imageFiles, imageFolderPath, exemplars, assignments)
%     num_clusters = length(exemplars);
%     
%     % Randomly select 5 exemplars if available
%     if num_clusters > 5
%         selected_exemplars = randsample(exemplars, 5);
%     else
%         selected_exemplars = exemplars;
%     end
% 
%     for cluster_idx = 1:length(selected_exemplars)
%         exemplar_idx = selected_exemplars(cluster_idx);
%         
%         % Find the images assigned to this exemplar
%         assigned_indices = find(assignments == exemplar_idx);
%         
%         % Check if the cluster has at least 3 images
%         if length(assigned_indices) >= 3
%             exemplar_image_path = fullfile(imageFolderPath, imageFiles(exemplar_idx).name);
%             exemplar_image = imread(exemplar_image_path);
%             
%             fprintf('Exemplar %d: %s\n', cluster_idx, imageFiles(exemplar_idx).name);
%             
%             % Randomly select 2 images from the cluster
%             random_indices = randsample(assigned_indices, 2);
%             
%             % Create a new figure
%             figure;
%             
%             % Display the exemplar
%             subplot(1, 3, 1);
%             imshow(exemplar_image);
%             title(sprintf('Exemplar %d: %s', cluster_idx, imageFiles(exemplar_idx).name));
%             
%             % Display the 2 random images
%             for i = 1:length(random_indices)
%                 assigned_image_path = fullfile(imageFolderPath, imageFiles(random_indices(i)).name);
%                 assigned_image = imread(assigned_image_path);
%                 
%                 fprintf('  Assigned image: %s\n', imageFiles(random_indices(i)).name);
%                 subplot(1, 3, i + 1);
%                 imshow(assigned_image);
%                 title(sprintf('Assigned image: %s', imageFiles(random_indices(i)).name));
%             end
%         end
%     end
% end

% function [exemplars, assignments] = cluster_with_affinity_propagation(features, desired_clusters)
%     % Define parameters for affinity propagation
%     max_iter = 200;
%     conv_iter = 15;
%     damping = 0.5;
%     
%     % Compute the affinity matrix
%     affinity_matrix = -pdist2(features, features, 'squaredeuclidean');
%     n = size(affinity_matrix, 1);
%     
%     % Initialize variables for the iterative approach
%     min_quantile_val = 0;
%     max_quantile_val = 1;
%     max_iterations = 20;
%     iteration = 0;
%     
%     % Iterate until the desired number of clusters is achieved or the maximum number of iterations is reached
%     while iteration < max_iterations
%         iteration = iteration + 1;
%         
%         % Set the preference value using the average of the min and max quantile values
%         quantile_val = (min_quantile_val + max_quantile_val) / 2;
%         preference = quantile(affinity_matrix(:), quantile_val);
%         
%         % Initialize matrices
%         availability = zeros(n, n);
%         responsibility = zeros(n, n);
%         tmp = zeros(n, n);
% 
%         % Affinity propagation iterations
%         for iter = 1:max_iter
%             % Update responsibility
%             tmp = affinity_matrix + availability;
%             I = eye(n);
%             tmp_max = max(tmp .* (1 - I), [], 2);
%             responsibility = (affinity_matrix - repmat(tmp_max, 1, n)) * (1 - damping) + responsibility * damping;
% 
%             % Update availability
%             tmp = max(0, responsibility);
%             availability_diag = sum(tmp, 2) - diag(tmp);
%             tmp = min(0, repmat(availability_diag, 1, n) + tmp);
%             availability = (tmp - tmp .* I) * (1 - damping) + availability * damping;
%         end
%         
%         % Find exemplars and assignments
%         criterion = responsibility + availability;
%         [~, exemplars] = max(criterion, [], 2);
%         [~, assignments] = max(criterion, [], 1);
%         exemplars = unique(exemplars);
%         
%         % Update the quantile values based on the current number of clusters
%         num_clusters = length(exemplars);
%         if num_clusters > desired_clusters
%             min_quantile_val = quantile_val;
%         elseif num_clusters < desired_clusters
%             max_quantile_val = quantile_val;
%         else
%             break;
%         end
%     end
% end
% 

% 
% function [cluster_centers, assignments] = cluster_with_kmeans(features, num_clusters)
%     % Perform k-means clustering
%     [assignments, cluster_centers] = kmeans(features, num_clusters, 'MaxIter', 500);
% end

% % Function to display exemplars and assigned images
% function display_exemplars_and_assigned_images(imageFiles, imageFolderPath, exemplars, assignments)
%     num_clusters = length(exemplars);
%     for cluster_idx = 1:num_clusters
%         exemplar_idx = exemplars(cluster_idx);
%         exemplar_image_path = fullfile(imageFolderPath, imageFiles(exemplar_idx).name);
%         exemplar_image = imread(exemplar_image_path);
%         
%         fprintf('Exemplar %d: %s\n', cluster_idx, imageFiles(exemplar_idx).name);
%         figure;
%         imshow(exemplar_image);
%         title(sprintf('Exemplar %d: %s', cluster_idx, imageFiles(exemplar_idx).name));
%         
%         % Find the images assigned to this exemplar
%         assigned_indices = find(assignments == exemplar_idx);
%         for assigned_idx = assigned_indices'
%             assigned_image_path = fullfile(imageFolderPath, imageFiles(assigned_idx).name);
%             assigned_image = imread(assigned_image_path);
%             
%             fprintf('  Assigned image: %s\n', imageFiles(assigned_idx).name);
%             figure;
%             imshow(assigned_image);
%             title(sprintf('Assigned image: %s', imageFiles(assigned_idx).name));
%         end
%     end
% end

%%
% % Function to cluster features using Affinity Propagation
% function [exemplars, assignments] = cluster_with_affinity_propagation(features)
%     features = double(features);
%     % Calculate the similarity matrix (negative squared Euclidean distance)
%     similarity_matrix = -pdist2(features, features, 'squaredeuclidean');
% 
%     % Compute the preference value (the median of the similarity values)
%     preference = median(similarity_matrix(:));
% 
%     % Perform Affinity Propagation clustering
%     [assignments, exemplars] = affinitypropagation(similarity_matrix, 'Preference', preference, 'SimilarityInput', true);
% end
% 
% % Function to display exemplars and one assigned image
% function display_exemplars_and_assigned_images(imageFiles, imageFolderPath, exemplars, assignments)
%     num_exemplars = numel(exemplars);
%     figure;
% 
%     for i = 1:num_exemplars
%         % Display the exemplar image
%         exemplar_path = fullfile(imageFolderPath, imageFiles(exemplars(i)).name);
%         exemplar_img = imread(exemplar_path);
%         subplot(num_exemplars, 2, (i - 1) * 2 + 1);
%         imshow(exemplar_img);
%         title(sprintf('Exemplar %d', i));
% 
%         % Find the indices of the images assigned to the current exemplar
%         assigned_indices = find(assignments == exemplars(i));
% 
%         % If there are assigned images, display the first one
%         if ~isempty(assigned_indices)
%             assigned_path = fullfile(imageFolderPath, imageFiles(assigned_indices(1)).name);
%             assigned_img = imread(assigned_path);
%             subplot(num_exemplars, 2, (i - 1) * 2 + 2);
%             imshow(assigned_img);
%             title(sprintf('Assigned Image for Exemplar %d', i));
%         end
%     end
% end
% 
% %%
% 
% function [labels, exemplars] = affinityPropagation(features, damping, max_iter)
% % Perform affinity propagation clustering on the given features.
% %
% % Inputs:
% %   features: An n-by-d matrix of n data points, each with d features.
% %   damping: A scalar value between 0 and 1 that controls the damping factor.
% %            Higher damping values lead to slower convergence but more
% %            stable results.
% %   max_iter: The maximum number of iterations to perform.
% %
% % Outputs:
% %   labels: An n-by-1 vector of cluster labels for each data point.
% %   exemplars: A k-by-d matrix of k exemplars, where k is the number of clusters.
% %
% % Written by ChatGPT, based on the MATLAB documentation.
% 
% % Calculate the similarity matrix
% S = pdist2(features, features, 'euclidean');
% S = -S.^2;
% 
% % Initialize messages
% n = size(features, 1);
% A = zeros(n, n);
% R = zeros(n, n);
% E = zeros(n, 1);
% 
% % Main loop
% for i = 1:max_iter
%     % Update responsibility matrix
%     Rp = bsxfun(@plus, S, E');
%     [Y, I] = max(Rp, [], 2);
%     for j = 1:n
%         Rp(j, I(j)) = -inf;
%     end
%     R = damping * R + (1 - damping) * Rp;
%     
%     % Update availability matrix
%     Aprev = A;
%     Rpos = max(R, 0);
%     for j = 1:n
%         A(j, :) = sum(Rpos([1:j-1,j+1:n], :), 1);
%         A(j, j) = sum(Rpos(j, [1:j-1,j+1:n]));
%     end
%     Adiag = diag(A);
%     A = min(A, 0);
%     for j = 1:n
%         A(j, j) = Adiag(j);
%     end
%     
%     % Update exemplars
%     Eprev = E;
%     E = diag(A + R);
%     
%     % Check for convergence
%     if all(A == Aprev)
%         if all(E == Eprev)
%             break;
%         end
%     end
% end
% 
% % Assign cluster labels
% labels = zeros(n, 1);
% for i = 1:n
%     if E(i) == max(A(i, :) + E')
%         labels(i) = i;
%     else
%         labels(i) = find(A(i, :) + E' == max(A(i, :) + E'), 1);
%     end
% end
% 
% % Identify exemplars
% exemplars = features(unique(labels), :);
% 
% end
% 
% function display_exemplars(imageFiles, imageFolderPath, exemplars, labels)
% % Display 6 random image exemplars for each cluster.
% %
% % Inputs:
% %   imageFiles: A struct array of file information for the images.
% %   imageFolderPath: The path to the folder containing the images.
% %   exemplars: A k-by-d matrix of k exemplars, where k is the number of clusters.
% %   labels: An n-by-1 vector of cluster labels for each data point.
% %
% % Written by ChatGPT.
% 
% % Get the unique cluster labels
% unique_labels = unique(labels);
%     
% % Randomly select 6 exemplars
% exemplar_idx = randperm(length(exemplars), min(6, length(exemplars)));
% 
% % Display the exemplar images
% for j = 1:length(exemplar_idx)
%     exemplar_path = fullfile(imageFolderPath, imageFiles(exemplar_idx(j)).name);
%     exemplar_image = imread(exemplar_path);
%     subplot(length(unique_labels), 6, (i-1)*6+j);
%     imshow(exemplar_image);
%     title(sprintf('Exemplar %d', j));
% end
% 
% end