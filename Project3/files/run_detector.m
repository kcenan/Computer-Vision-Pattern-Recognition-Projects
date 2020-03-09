% Starter code prepared by James Hays
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

for i = 1:length(test_scenes)
      
    %fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    count=1;
    cur_bboxes=[];
    cur_confidences=[];
    cur_image_ids=cell(0, 1);
    D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
    boundary=6;
    threshold = -1.2; %setting the optimal thresold for the scores
    max_window_scale = 6; %maximum different type sliding window
    for j=1:max_window_scale %we itarate al the scales whrough the give image to detect the differentt sized faces
        slidingwindow=8*j; %size of the sliding window
        hog = vl_hog(img, slidingwindow); %getting histogram of gradient for the image
        %we are setting boundry so that we don2t get out of indices of
        %matrix
        y_boundary = size(hog, 1)-(boundary);
        x_boundary = size(hog, 2)-(boundary);
        %itarating through all the pixels of the image until the boundary
        %is reached
        for k=1:y_boundary
            for l=1:x_boundary
                cur_top_y = k;%y point of reference point
                cur_left_x = l;%x point of reference point
                coor_matrix=[cur_left_x, cur_top_y, cur_left_x+boundary-1, cur_top_y+boundary-1]; % a vector which holds the coordnates of the current fame
                sized_coor_matrix = coor_matrix * slidingwindow; %sizing the coordinate matrix
                %to fix a small bug occured in our code to prevent it from
                %going out of indices
                if y_boundary == size(hog, 1)
                 y_boundary = size(hog, 1)-(boundary);
                end
                if x_boundary == size(hog, 2)
                 x_boundary = size(hog, 2)-(boundary);
                end
                
                frame = hog(cur_top_y:cur_top_y+boundary-1, cur_left_x:cur_left_x+boundary-1, :); %getting the local frame to check whether it is a face or not
                cur_data = reshape(frame, 1, D); %making the frame a vector so that we an use it to calculate the score
                score = w'*cur_data'+b; %we got this fomula of score from the cl_svmtrain, it calculates the score of the given frame by using the trained classifier
                 %if score is below the treshold we go to the next
                 %itaration
                if (score <= threshold)
                  continue  
                end
                    cur_bboxes= vertcat(cur_bboxes, sized_coor_matrix); %adding the current sized coordinate matrix of the frame f it exceeds the ttreshold
                    cur_confidences= vertcat(cur_confidences, score); %putting the score to the matrix of confidences
                    cur_image_ids{count,1}=test_scenes(i).name;
                    count = count+1; %if the count is equal to 1 it means we are done whit this image and we are iterating to the next image
                    %if there is a frame which has greater value than the
                    %treshold the counter will be more than one which means
                    %we will do non maximum suppressing
            end
        end
    end
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    if count ~= 1
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
    end
end


