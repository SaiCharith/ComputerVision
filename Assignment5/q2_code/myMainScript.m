%% Read Video & Setup Environment
clear
clc
close all hidden
[FileName,PathName] = uigetfile({'*.avi'; '*.mp4'},'Select shaky video file');

cd mmread
vid=mmread(strcat(PathName,FileName));
cd ..
s=vid.frames;

%% Your code here
%display(s);
%displayvideo(s,0.01);

T = size(s,2);
vector_of_transforms=[];
vector_of_transforms_x=[];

vector_of_transforms_y=[];
vector_of_transforms_theta=[];

vector_of_transforms_scale=[];

for i=1:T-1
  prev_frame=rgb2gray(s(i).cdata);
  curr_frame=rgb2gray(s(i+1).cdata);
  prev_frame_points=detectSURFFeatures(prev_frame);
  curr_frame_points=detectSURFFeatures(curr_frame);
%   figure; imshowpair(prev_frame,curr_frame,'ColorChannels','red-cyan');
  [f1,vpts1] = extractFeatures(prev_frame,prev_frame_points);
  [f2,vpts2] = extractFeatures(curr_frame,curr_frame_points);
  indexPairs = matchFeatures(f1,f2) ;
  
  matchedPoints1 = vpts1(indexPairs(:,1));
  matchedPoints2 = vpts2(indexPairs(:,2));
  
  
  
%   figure; showMatchedFeatures(prev_frame, curr_frame, matchedPoints1, matchedPoints2);
%   legend('A', 'B');
%     transformType='similarity';
    
    transformType='affine';
%     transformType='projective';
  [tform,inlier1,inlier2] = estimateGeometricTransform(matchedPoints1,matchedPoints2,transformType);
    H=ransacHomography(matchedPoints1,matchedPoints2,5);
    %   display(tform);
        vector_of_transforms=[vector_of_transforms affine2d(H)];

    H = tform.T;
    R = H(1:2,1:2);
    % Compute theta from mean of two possible arctangents
    theta = mean([atan2(R(2),R(1)) atan2(-R(3),R(4))]);
    % Compute scale from mean of two stable mean calculations
    scale = mean(R([1 4])/cos(theta));
    % Translation remains the same:
    translation = H(3, 1:2);
    % Reconstitute new s-R-t transform:
%     HsRt = [[scale*[cos(theta) -sin(theta); sin(theta) cos(theta)]; ...
%       translation], [0 0 1]'];
%     tformsRT = affine2d(HsRt);
    vector_of_transforms_x=[vector_of_transforms_x H(3,1)];
  
  vector_of_transforms_y=[vector_of_transforms_y H(3,2)];
  
  vector_of_transforms_theta=[vector_of_transforms_theta theta];

  vector_of_transforms_scale=[vector_of_transforms_scale scale];
end
% display(vector_of_transforms);

halfwindow=15;


noisy_sequence_x=vector_of_transforms_x;

noisy_sequence_scale=vector_of_transforms_scale;

noisy_sequence_y=vector_of_transforms_y;

noisy_sequence_theta=vector_of_transforms_theta;
for i=1:T-1
       lower=max(1,i-halfwindow);
       upper=min(T-1,i+halfwindow);
       x=0;
       y=0;
       theta=0;
       scale=0;
       for j=lower:upper
           x=x+vector_of_transforms_x(j);
           
           y=y+vector_of_transforms_y(j);
           
           theta=theta+vector_of_transforms_theta(j);
           
          scale=scale+vector_of_transforms_scale(j);
       end
       x=x/(upper-lower+1);
       y=y/(upper-lower+1);
       scale=scale/(upper-lower+1);
       theta=theta/(upper-lower+1);
%        theta=0;
%        scale=1;
       matrix = [[scale*[cos(theta) -sin(theta); sin(theta) cos(theta)]; x y], [0 0 1]'];
%        vector_of_transforms=[vector_of_transforms affine2d(matrix)];
       
       vector_of_transforms_x(i)=x;
       
       vector_of_transforms_y(i)=y;
       
       vector_of_transforms_theta(i)=theta;
       
       
       vector_of_transforms_scale(i)=scale;
end

figure;
plot(noisy_sequence_x);
hold on;
plot(vector_of_transforms_x);
legend('noisy','smoothed')
title('for translation in x');

figure;
plot(noisy_sequence_y);
hold on;
plot(vector_of_transforms_y);
legend('noisy','smoothed')
title('for translation in y');


figure;
plot(noisy_sequence_theta);
hold on;
plot(vector_of_transforms_theta);
legend('noisy','smoothed')
title('for translation in theta');


figure;
plot(noisy_sequence_scale);
hold on;
plot(vector_of_transforms_scale);
legend('noisy','smoothed')
title('for translation in scale');


H=size(s(1).cdata,1);

W=size(s(1).cdata,2);

outV=s;

for j=2:T
    out=imwarp(outV(j-1).cdata,vector_of_transforms(j-1),'OutputView',imref2d(size(s(j-1).cdata)));
    outV(j).cdata=out(1:H,1:W,:);
end
figure;
displayvideo(outV,0.005);








N=T-1;




%% Write Video
vfile=strcat(PathName,'combined_',FileName);
ff = VideoWriter(vfile);
ff.FrameRate = 30;
open(ff)

for i=1:N+1
    f1 = s(i).cdata;
    f2 = outV(i).cdata;
    vframe=cat(1,f1, f2);
    writeVideo(ff, vframe);
end
close(ff)

%% Display Video
figure
msgbox(strcat('Combined Video Written In ', vfile), 'Completed') 
displayvideo(outV,0.01)
