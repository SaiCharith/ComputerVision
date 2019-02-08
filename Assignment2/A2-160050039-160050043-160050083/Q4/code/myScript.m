function [superImage] = myScript(imageName1, imageName2, imageName3, threshold)
    % Reference:    https://in.mathworks.com/help/vision/ref/matchfeatures.html
    origI1 = imread(strcat('../input/', imageName1)); % Original Image 1
    origI2 = imread(strcat('../input/', imageName2)); % Original Image 2
    origI3 = imread(strcat('../input/', imageName3)); % Original Image 3
    I1 = rgb2gray(origI1);
    I2 = rgb2gray(origI2);
    I3 = rgb2gray(origI3);
    
    points1 = detectSURFFeatures(I1);
    points2 = detectSURFFeatures(I2);
    points3 = detectSURFFeatures(I3);
    [f1,vpts1] = extractFeatures(I1,points1);
    [f2,vpts2] = extractFeatures(I2,points2);
    [f3,vpts3] = extractFeatures(I3,points3);
    indexPairs1 = matchFeatures(f1,f2);
    indexPairs3 = matchFeatures(f2,f3);
    
    % Finding Matching Pixel Correspondences

    % SURF is exchanging x & y that's why
    a = vpts1(indexPairs1(:,1)).Location;
    matchedPoints1x = a(:,1);
    matchedPoints1y = a(:,2);
    matchedPoints1 = [matchedPoints1y matchedPoints1x];

    b = vpts2(indexPairs1(:,2)).Location;
    matchedPoints2x = b(:,1);
    matchedPoints2y = b(:,2);
    matchedPoints2 = [matchedPoints2y matchedPoints2x];
    
    % Ransac Homography such that matchedPoints2 = H * matchedPoints1
    H1 = ransacHomography(matchedPoints1, matchedPoints2, threshold);
    
    a = vpts2(indexPairs3(:,1)).Location;
    matchedPoints1x = a(:,1);
    matchedPoints1y = a(:,2);
    matchedPoints1 = [matchedPoints1y matchedPoints1x];
    
    b = vpts3(indexPairs3(:,2)).Location;
    matchedPoints2x = b(:,1);
    matchedPoints2y = b(:,2);
    matchedPoints2 = [matchedPoints2y matchedPoints2x];
    
    H3 = ransacHomography(matchedPoints1, matchedPoints2, threshold);
    
    % Initializing stitched Image with a super Image containing original I1 at
    % its centre
    height = max(size(I1, 1), size(I2, 1));
    width = max(size(I1, 2), size(I2, 2));
    superImage = uint8(zeros(3*height, 3*width, 3));
    superImage(height+1:2*height, width+1:2*width, :) = origI2(:, :, :);
    
    % Reverse Warping
    for i=1:3*height
        for j=1:3*width
            if (~((i >= height + 1) && (i <= 2*height) && (j >= width + 1) && (j <= 2*width)))
                coord = H1\[(i-height); (j-width); 1];
                coord = coord./coord(3);
                x = floor(coord(1));
                y = floor(coord(2));
                if ((x <= size(I1, 1)) && (x >= 1) && (y <= size(I1, 2)) && (y >= 1))
                    % Valid Point in I1 exists
                    superImage(i,j,:) = origI1(x,y,:);
                end
                
                coord = H3*[(i-height); (j-width); 1];
                coord = coord./coord(3);
                x = floor(coord(1));
                y = floor(coord(2));
                if ((x <= size(I3, 1)) && (x >= 1) && (y <= size(I3, 2)) && (y >= 1))
                    % Valid Point in I3 exists
                    superImage(i,j,:) = origI3(x,y,:);
                end
            end
        end
    end
end