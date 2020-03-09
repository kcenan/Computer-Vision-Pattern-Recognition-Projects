% Local Feature Stencil Code
% CS 143 Computater Vision, Brown U.
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the features as additional inputs.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features 1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features(features1, features2)

% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.

% Placeholder that you can delete. Random matches and confidences


distance_matrix = zeros(size(features1,1),size(features2,1));

%find the distances between vectors
for i= 1:size(features1,1)
    for j = 1:size(features2,1)
        
        distance_matrix(i,j) = sqrt( sum( (features1(i,:)- features2(j,:)).^2  ));
        
    end
end

threshold = 0.7;
%I sorted distance matrix
[sorted_dist_matrix, second_image_indexes] = sort(distance_matrix, 2);

%I made the ratio test
ratios = sorted_dist_matrix(:,1)./sorted_dist_matrix(:,2);
ratio_test_passed_vectors = ratios < threshold;
confidences = 1./ratios( ratio_test_passed_vectors );

matches = zeros(size(confidences,1), 2);
matches(:,1) = find(ratio_test_passed_vectors);
matches(:,2) = second_image_indexes(ratio_test_passed_vectors, 1);

% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
[confidences, ind] = sort(confidences, 'descend');
matches = matches(ind,:);