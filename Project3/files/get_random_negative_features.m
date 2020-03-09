% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

    image_files = dir( fullfile( non_face_scn_path, '*.jpg' )); %extracting non face images
    N = length(image_files);%number of nonface images
    D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31; 
    samplePerImage =  ceil(num_samples/N); %calculating for each image how many sample we will pick
    difference = (- num_samples/N + ceil(num_samples/N)) * N; %we have more samples because we used ceil function we are deleting extra samples
    totalNumFeatures = N * samplePerImage - difference; %total number feature we will use in function
    features_neg = zeros([totalNumFeatures D]); 
   
    sampleCount=1; %we are counting samples when we rach totalnumfeature we will finish the function
    for j=1:N %we are extracting imaes one by one
       get_name = image_files(j).name; %exracting name of current image
       current_image = imread(get_name);
       current_image = rgb2gray(current_image); %changiing the image to grey as expected in assignment
       current_image = im2single(current_image); %Convert image to single precision.  
      %taking width and height of images to use later
       sizeImage = size(current_image); 
       imageWidth = sizeImage(1);
       imageHeight = sizeImage(2);
       frameSize = feature_params.template_size;
       for i = 1:samplePerImage %for each sample we will add it as a negative feature
           %random reference point but we are subtracting frame size to
           %inhibt exceeding matrix
           leftSidex = ceil(rand() * (imageWidth-frameSize)); 
           leftSidey = ceil(rand() * (imageHeight-frameSize));
           frame = current_image(leftSidex:leftSidex+frameSize-1,leftSidey:leftSidey+frameSize-1); %matrix of frame which we will take as an sample
           currentFrameHog = vl_hog(frame, feature_params.hog_cell_size); %histogram of gradient of frame
           sampleCount = sampleCount+1; %incrementing sample count
           samplesLoc = (j-1)*samplePerImage + i; %we are adding all samples to on samples .So we are memorasing samples location
           currentFrameHogColumn = reshape(currentFrameHog, 1, D); %we need 1xD matrix so we are reshaping current matrix
           features_neg(samplesLoc,:) = currentFrameHogColumn; %adding current frame hog collumns to output
            if sampleCount == num_samples %if we reach num_samples we are terminating function
                return
            end
       end
      end
    end