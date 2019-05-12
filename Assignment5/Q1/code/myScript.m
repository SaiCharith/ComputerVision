clear;
close all;
clc;
%% Your Code Here
[status, msg, msgID] = mkdir('../output');  % Because output images get saved in ../output
%% 1a)
I = imread(strcat("../input/", num2str(1), ".jpg"));
featureType = @detectSURFFeatures;  % Type of features
featurePoints = featureType(I);

figure; imshow(I); hold on; plot(featurePoints);
pause(2);
saveas(gcf, "../output/All.jpg");

%% 1b)

TEMPLATE_SIZE = 3;  % (2*3+1) X (2*3+1) = 7 X 7 Window
NUM_OF_POINTS = 4;  % Number of salient features to extract
salientFeaturePoints = getSalientFeaturePoints(featureType, I, NUM_OF_POINTS, TEMPLATE_SIZE);
myplot(salientFeaturePoints.Location(:,1), salientFeaturePoints.Location(:,2), I, 0, NUM_OF_POINTS);   % View the salientfeaturepoints returned by the function
pause(2);

%% 1 (c,d,e)
NUM_OF_FRAMES = 247;  % It starts reading from frame 1 till NUM_OF_FRAMES excluding frame 60 which is not present
templateFrame = double(I);

trackedLocations = salientFeaturePoints.Location;   % It stores locations of feature points for all frames so far
myplot(salientFeaturePoints.Location(:,1), salientFeaturePoints.Location(:,2), templateFrame, 1, NUM_OF_POINTS);   % View the first frame

% For every frame
for t=2:NUM_OF_FRAMES
    if mod(t,25) == 0
        fprintf("Closing figures to save memory\n");
        close all;  % To avoid going to swap space
    end
    if t == 60  % Because frame 60 is absent
        continue;
    end
    
    if mod(t,10) == 0   % After every 10 frames change template frame
        sizeTracked = size(trackedLocations, 1);
        salientFeaturePoints.Location = trackedLocations(sizeTracked-NUM_OF_POINTS+1:sizeTracked, :);
        templateFrame = double(imread(strcat("../input/", num2str(t-1), ".jpg")));
    end
    
    frame = double(imread(strcat("../input/", num2str(t), ".jpg")));
    [Ix, Iy] = imgradientxy(imgaussfilt(frame,2)); % Calculate gradients
%     Ix = imgaussfilt(Ix,1);
%     Iy = imgaussfilt(Iy,1);

    for k=1:NUM_OF_POINTS   % For each feature point
        point = salientFeaturePoints.Location(k,:);
        
        % Estimate motion and calculate L2 error       
        [motion, L2error] = estimateMotion(templateFrame, frame, point, TEMPLATE_SIZE, Ix, Iy);
        
        newLocation = motion*[point'; 1];
        trackedLocations = [trackedLocations; newLocation'];
        fprintf("L2 Error for frame no. %d, & point no. %d, is = %d\n", t, k, L2error);
    end
    myplot(trackedLocations(:, 1), trackedLocations(:, 2), frame, t, NUM_OF_POINTS);
end
close all;

function [] = myplot(xCoords, yCoords, image, index, np)
    % Caller should not swap points as required for plot
    % Assuming four different type of points are there    
    % xCoords and yCoords should be column vectors    
    figure; imshow(uint8(image)); hold on;
    s = size(yCoords, 1);
    colors = ["c*", "y*", "m*", "g*", "b*", "r*", "k*", "w*"];
    for i=1:np
        plot(yCoords(i:np:s), xCoords(i:np:s), colors(mod((i-1), 8) + 1), 'MarkerSize', 8, 'LineWidth', 1);
    end
    saveas(gcf, strcat("../output/", num2str(index), ".jpg"));
end