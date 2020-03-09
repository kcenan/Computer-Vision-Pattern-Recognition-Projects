% Local Feature Stencil Code
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or(b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width)


image = im2double(image);
gaussian = fspecial('gaussian',feature_width,1);
image = imfilter(image,gaussian);

sobelx = [1 0 -1;2 0 -2;1 0 -1];
sobely = sobelx';
%it is the derivative of image up to x
Dx=imfilter(image,sobelx);
%it is the derivative of image up to y
Dy=imfilter(image,sobely);

Dx2 = Dx.^2;
Dy2 = Dy.^2;
Dxy = Dx.*Dy;
%another gaussian which makes the image more detectable

sigma = 8;
h_size= 5;
gaussian2 = fspecial('gaussian',h_size,sigma);
Dx2 = imfilter(Dx2,gaussian2);
Dy2 = imfilter(Dy2,gaussian2);
Dxy = imfilter(Dxy,gaussian2);

%this formula gives the harris measure which helps us to find interest
%points
k = 0.05;
harris_measure = (Dx2.*Dy2 - Dxy.^2) - k *(Dx2+Dy2);

%local max bulup sap alta koyucaz
radius = 1;
path_lenght = 2 * radius+ 1 ;
max_points = ordfilt2(harris_measure,path_lenght.^2,ones(path_lenght));

[k1,k2] = size(harris_measure);
treshold = 0.005;

%we are checking the point which we detected is correct for treshold which
%we determined and check it whether it is liable to max_points which we
%constructed before

for i = 1:k1
    for j = 1:k2
     if harris_measure(i,j) == max_points(i,j)  &&  harris_measure(i,j)>treshold
        edges(i,j) = 1;
     end
    end
end

edges = edges(feature_width   : k1-feature_width  , feature_width   : k2-feature_width);
edges = padarray(edges,[feature_width ,feature_width]);
%x and y show us the corner points
[x,y] = find(edges);   



% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.


end