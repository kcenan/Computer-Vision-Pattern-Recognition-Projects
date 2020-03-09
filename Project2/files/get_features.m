% Local Feature Stencil Code
% Written by James Hays
 
% Returns a set of feature descriptors for a given set of interest points. 
 
% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.
 
% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)
 
function [features] = get_features(image, x1, y1, feature_width)
 
% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See lecture notes or Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)
 
% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% Histogram: You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.
 
 
% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

    
    features = zeros(size(x1,1), 128);
    image = im2double(image);
    %gaussian filter to smooth to reach better results
    gaussian_small = fspecial ('gaussian',feature_width , 0.5);
    gaussian_large = fspecial ('gaussian',feature_width , 8);
 
    %derivatives of image
    image = conv2(image , gaussian_small, 'same');
    %imgradient gives us each pixels magnitudes and gradient direction
    [magnitudes , directions] = imgradient(image);
      
    for i = 1: size(x1,1)
    %taking 16x16 array which covers the feature point
    magnitudepart = magnitudes(x1(i)-8: x1(i)+7,y1(i)-8:y1(i)+7);
    magnitudepart = magnitudepart .* gaussian_large;
    directionpart = directions(x1(i)-8: x1(i)+7,y1(i)-8:y1(i)+7);
    directionpart = directionpart .* gaussian_large;
    
    %dividing 16x16 matrix to 16 4x4 matrixes
    smallmag= mat2cell(magnitudepart, [4,4,4,4], [4,4,4,4]);
    smalldir= mat2cell(directionpart, [4,4,4,4], [4,4,4,4]);
    
    a=1;
    for j = 1:4
        for k = 1:4
            
            min_mag_patch = smallmag{j,k};
            min_dir_patch = smalldir{j,k};
            for d = 1:4
                for h = 1:4
                    for m = -4:3
                        if (min_dir_patch(d,h) < (m+1)*45 && min_dir_patch(d,h) >= m*45)  
                      %each array part is checked to add to histogram array
                      %each collumn has histogram array of each feature
                      %point which include 16(array part)x8(direction part comes from 16 variables) = 128 rows
                        features(i,(a-1)*8 + m+5) = features(i,(a-1)*8 + m+5) + min_mag_patch(d,h);               
                        end
                    end
                end
            end
            
            a = a + 1;
        end
    end
    end
  
    
    %I normilised the feature vector
 features = normc(features);
    
end