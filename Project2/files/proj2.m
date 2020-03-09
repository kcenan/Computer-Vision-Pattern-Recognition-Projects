% Local Feature Stencil Code
% Written by James Hays

% This script 
% (1) Loads and resizes images
% (2) Finds interest points in those images                 (you code this)
% (3) Describes each interest point with a local feature    (you code this)
% (4) Finds matching features                               (you code this)
% (5) Visualizes the matches
% (6) Evaluates the matches based on ground truth correspondences

close all

% You can simply download images off the Internet, as well. However, the
% evaluation function at the bottom of this script will only work for this
% particular image pair (unless you add ground truth annotations for other
% image pairs). It is suggested that you only work with these two images
% until you are satisfied with your implementation and ready to test on
% additional images. A single scale pipeline works fine for these two
% images (and will give you full credit for this project), but you will
% need local features at multiple scales to handle harder cases.
image1 = imread('image1.jpg');
image2 = imread('image2.jpg');

% You don't have to work with grayscale images. Matching with color
% information might be helpful.
image1 = rgb2gray(single(image1)/255);
image2 = rgb2gray(single(image2)/255);

scale_factor = 0.5; %make images smaller to speed up the algorithm
image1 = imresize(image1, scale_factor, 'bilinear');
image2 = imresize(image2, scale_factor, 'bilinear');

feature_width = 16; %width and height of each local feature, 16x16, in pixels. 

%% Find distinctive points in each image. 
% !!! You will need to implement get_interest_points. !!!
[x1, y1] = get_interest_points(image1, feature_width);
[x2, y2] = get_interest_points(image2, feature_width);

figure, imshow(image1), hold on;
plot (y1,x1, 'ys');

figure, imshow(image2), hold on;
plot (y2,x2, 'ys');
%% Create feature vectors at each interest point. 
% !!! You will need to implement get_features. !!!
[image1_features] = get_features(image1, x1, y1, feature_width);
[image2_features] = get_features(image2, x2, y2, feature_width);


%% Match features. 
% !!! You will need to implement get_features. !!!
[matches, confidences] = match_features(image1_features, image2_features);

% You might want to set 'num_pts_to_visualize' and 'num_pts_to_evaluate' to
% some constant once you start detecting hundreds of interest points,
% otherwise things might get too cluttered. You could also threshold based
% on confidence.
num_pts_to_visualize = size(matches,1);
show_correspondence(image1, image2, y1(matches(1:num_pts_to_visualize,1)), ...
                                    x1(matches(1:num_pts_to_visualize,1)), ...
                                    y2(matches(1:num_pts_to_visualize,2)), ...
                                    x2(matches(1:num_pts_to_visualize,2)));

num_pts_to_evaluate = size(matches,1);
% All of the coordinates are being divided by scale_factor because of the
% imresize operation at the top of this script. This evaluation function
% will only work for the particular Notre Dame image pair specified in the
% starter code. You can, however, use 'collect_ground_truth_corr.m' to
% build ground truth for additional image pairs and then change the paths
% in 'evaluate_correspondence' accordingly. Or you can simply comment out
% this function once you start testing on additional image pairs.
evaluate_correspondence(x1(matches(1:num_pts_to_evaluate,1))/scale_factor, ...
                        y1(matches(1:num_pts_to_evaluate,1))/scale_factor, ...
                        x2(matches(1:num_pts_to_evaluate,2))/scale_factor, ...
                        y2(matches(1:num_pts_to_evaluate,2))/scale_factor);












