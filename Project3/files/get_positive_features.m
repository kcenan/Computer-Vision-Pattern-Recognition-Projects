% Starter code prepared by James Hays
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples.

function features_pos = get_positive_features(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.


%%%%%%%%%%%%%%%%%%%
%extracting face images
image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
N = length(image_files); %how many images do we have
D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31; %it is given number.we will get numerous random samples from each image which are 36x36. 
features_pos = zeros(N, D); %making an array which is NxD size as question sad
for img = 1:N  %we will get histogram of gradient from each photo
    get_name = fullfile(train_path_pos, image_files(img).name); %extracting names from directory
    current_image = imread(get_name);
    current_image = im2single(current_image); %we need to convert image to single precision to use in vl_hog function
    hog = vl_hog(current_image, feature_params.hog_cell_size); %it tooks histogram of gradient from current photo
    features_pos(img,:) = reshape(hog, 1, D); %we made 1xD matrix
end
    
%%%%%%%%%%%%%%%

% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray
