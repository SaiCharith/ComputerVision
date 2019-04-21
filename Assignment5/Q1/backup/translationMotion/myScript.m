clear;
close all;
clc;
%% Your Code Here
[status, msg, msgID] = mkdir('../ourOutput');  % Because output images get saved in ../ourOutput

NUM_OF_FRAMES = 24;  % It starts reading from frame 1 till NUM_OF_FRAMES
TEMPLATE_SIZE = 3;  % (2*3+1) X (2*3+1) = 7 X 7 Window
NUM_OF_POINTS = 60;  % Number of salient features to extract

I = imread(strcat("../input/", num2str(1), ".jpg"));
featureType = @detectHarrisFeatures;  % Type of features
featurePoints = featureType(I);
figure; imshow(I); hold on; plot(featurePoints.selectStrongest(NUM_OF_POINTS));
saveas(gcf, "../ourOutput/All.jpg");

salientFeaturePoints = getSalientFeaturePoints(featureType, I, NUM_OF_POINTS, TEMPLATE_SIZE);
% myplot(salientFeaturePoints.Location(:,1), salientFeaturePoints.Location(:,2), I, 0, NUM_OF_POINTS);   % View the salientfeaturepoints returned by the function

templateFrame = double(I);

trackedLocations = salientFeaturePoints.Location;   % It stores locations of feature points for all frames so far
myplot(salientFeaturePoints.Location(:,1), salientFeaturePoints.Location(:,2), templateFrame, 1, NUM_OF_POINTS);   % View the first frame

for t=2:NUM_OF_FRAMES
%     Changing frame every time
    templateFrame = double(imread(strcat("../input/", num2str(t-1), ".jpg")));
    sizeTracked = size(trackedLocations, 1);
    salientFeaturePoints.Location = trackedLocations(sizeTracked-NUM_OF_POINTS+1:sizeTracked, :);
%     if mod(t,6) == 0
%         %   Change the template frame and the location of salient features
%         sizeTracked = size(trackedLocations, 1);
%         salientFeaturePoints.Location = trackedLocations(sizeTracked-NUM_OF_POINTS+1:sizeTracked, :);
%         templateFrame = double(imread(strcat("../input/", num2str(t-1), ".jpg")));
%     end
    frame = double(imread(strcat("../input/", num2str(t), ".jpg")));
    if NUM_OF_POINTS == 0
        break;
    end
    
    [Ix, Iy] = imgradientxy(frame);
    threshold = TEMPLATE_SIZE*1000;
    count = 0;

    for k=1:NUM_OF_POINTS
        point = salientFeaturePoints.Location(k-count, :);
%         point = [point(2), point(1)];   % Checking with swapping
        [motion, L2error] = estimateMotion(templateFrame, frame, point, TEMPLATE_SIZE, Ix, Iy);   % TODO try (nice)swapping point here and (bad)smoothing
%         point = [point(2), point(1)];   % Checking with swapping
        newLocation = motion*[point'; 1];
        %   Removing bad points, If L2 error is more than threshold than discard that point
        %   Discarding includes changing salientfeatures and reducing num of points
        if L2error > threshold || newLocation(1) < 0 || newLocation(2) < 0
            fprintf("Rejected ");
            salientFeaturePoints = [salientFeaturePoints(1:k-count-1); salientFeaturePoints(k-count+1:NUM_OF_POINTS-count)];
            count = count + 1;
        else
            trackedLocations = [trackedLocations; newLocation'];
        end
        fprintf("L2 Error for frame no. %d, & point no. %d, is = %d\n", t, k, L2error);
    end
    NUM_OF_POINTS = NUM_OF_POINTS - count;
    sizeTracked = size(trackedLocations, 1);
    myplot(trackedLocations(sizeTracked-NUM_OF_POINTS+1:sizeTracked, 1), trackedLocations(sizeTracked-NUM_OF_POINTS+1:sizeTracked, 2), frame, t, NUM_OF_POINTS);
%     myplot(trackedLocations(:, 1), trackedLocations(:, 2), frame, t, NUM_OF_POINTS);
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
    saveas(gcf, strcat("../ourOutput/", num2str(index), ".jpg"));
end