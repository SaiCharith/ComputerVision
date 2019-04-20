clear;
close all;
clc;

%% Your Code Here
% 1a)
I = imread("../input/1.jpg");
featurePoints = detectSURFFeatures(I);
figure;
imshow(I); hold on;
plot(featurePoints);

% 1b)
% Image gradients of first frame using sobel operator
TEMPLATE_SIZE = 4; % 7 X 7 Window
NUM_OF_POINTS = 4;
salientFeaturePoints = getFeaturePoints(I, NUM_OF_POINTS, TEMPLATE_SIZE);
figure;
imshow(I); hold on;
plot(salientFeaturePoints);
pause(1);

% REST
NUM_OF_FRAMES = 39;
FIRST_FRAME = 1;
templateFrame = double(imread(strcat("../input/", num2str(FIRST_FRAME), ".jpg")));

allLocations = zeros(NUM_OF_FRAMES*NUM_OF_POINTS, 2);
allLocations((FIRST_FRAME-1)*NUM_OF_POINTS+1:FIRST_FRAME*NUM_OF_POINTS, :) = salientFeaturePoints.Location;
figure; imshow(uint8(templateFrame)); hold on;  plot(salientFeaturePoints.Location(:,1), salientFeaturePoints.Location(:,2), '*');
[status, msg, msgID] = mkdir('ourOutput');  saveas(gcf, strcat("../ourOutput/", num2str(FIRST_FRAME), ".jpg"));

for t=FIRST_FRAME+1:NUM_OF_FRAMES
    if mod(t,10) == 0
%         templateFrame = double(imread(strcat("../input/", num2str(t-1), ".jpg")));
    end
    frame = double(imread(strcat("../input/", num2str(t), ".jpg")));
    newLocation = zeros(NUM_OF_POINTS, 2);
    [Ix, Iy] = imgradientxy(frame);
    for k=1:NUM_OF_POINTS
        point = round(salientFeaturePoints.Location(k,:));
        point = [point(2) point(1)];
        motion = estimateMotion(templateFrame, frame, point, TEMPLATE_SIZE, Ix, Iy);
        newLocation(k, :) = motion*[point'; 1];
    end
    newLocation = [newLocation(:,2) newLocation(:,1)];
    allLocations((t-1)*NUM_OF_POINTS+1:t*NUM_OF_POINTS, :) = newLocation;
    figure; imshow(uint8(frame)); hold on;  plot(allLocations(:, 1), allLocations(:, 2), '*');
    saveas(gcf, strcat("../ourOutput/", num2str(t), ".jpg"));
end
close all;
% load gong.mat;  sound(y);
%% Save all the trajectories frame by frame
% variable trackedPoints assumes that you have an array of size 
% No of frames * 2(x, y) * No Of Features
% noOfFeatures is the number of features you are tracking
% Frames is array of all the frames(assumes grayscale)
% noOfPoints = 1;
% for i=1:N
%     NextFrame = Frames(i,:,:);
%     imshow(uint8(NextFrame)); hold on;
%     for nF = 1:noOfFeatures
%         plot(trackedPoints(1:noOfPoints, 1, nF), trackedPoints(1:noOfPoints, 2, nF),'*')
%     end
%     hold off;
%     saveas(gcf,strcat('output/',num2str(i),'.jpg'));
%     close all;
%     noOfPoints = noOfPoints + 1;
% end 
%    