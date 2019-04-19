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
pause(1);
close;

% 1b)
% Image gradients of first frame using sobel operator
TEMPLATE_SIZE = 3; % 7 X 7 Window
NUM_OF_POINTS = 4;
salientFeaturePoints = getFeaturePoints(I, NUM_OF_POINTS, TEMPLATE_SIZE);
figure;
imshow(I); hold on;
plot(salientFeaturePoints);
pause(1);
close;

% 1c)
NUM_OF_FRAMES = 127;
templateFrame = im2double(imread(strcat("../input/", num2str(1), ".jpg")));
[Ix, Iy] = imgradientxy(templateFrame);

for t=2:NUM_OF_FRAMES
    frame = im2double(imread(strcat("../input/", num2str(t), ".jpg")));
    newLocation = zeros(NUM_OF_POINTS, 2);
    for k=1:NUM_OF_POINTS
        point = round(salientFeaturePoints.Location(k,:));
        motion = estimateMotion(templateFrame, frame, point, TEMPLATE_SIZE, Ix, Iy);
        newLocation(k, :) = motion*[point'; 1];
    end
    figure;
    imshow(frame); hold on;
    plot(round(newLocation(:, 1)), round(newLocation(:, 2)), '*');
    savefig(strcat("../ourOutput/", num2str(t), ".fig"));
    pause(4);
    close;
end

% figure;
% imshowpair(Ix,Iy,'montage')
% title('Directional Gradients Gx and Gy, Using Sobel Method')

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
