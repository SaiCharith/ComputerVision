function [ H ] = ransacHomography( x1, x2, thresh )
% Returning a "good" homography between x1 & x2 such that x2 = H*x1;
%% 
% CONSTANTS
p = 0.4; % Assuming 40% inliers
k = 4; % Number of corresponding points is 4
N = size(x1, 1); % Total number of correspondences
S = ceil(p^-k); % Expected number of subsets to be taken to get all inliers
maxConsensusCount = 0; % 
H = zeros(3,3);
 
% Preprocessing x1 and x2 to convert them into homogeneous coordinates
x1 = [x1 ones(N,1)];
x2 = [x2 ones(N,1)];

% Doing Homography S number of times where S is the expected number of 
% subsets for getting all inliers in atleast one subset
for i = 1:10*S    
    % k random indices from 1-N without repitition
    indices = datasample([1:N], k, 'Replace', false);
    
    % Corresponding Data Points from x1 & x2 
    p1 = x1(indices, :);
    p2 = x2(indices, :);
    
    % Estimating Homography such that p2 = H*p1
    tempH = homography(p1, p2);
    
    % Counting the size of consensus set, It doesn't matter if also include
    % C while calculating consesus set
    calculatedX2 = x1*tempH';
    calculatedX2 = calculatedX2./calculatedX2(:,3);
    ConsensusCount = sum(sum((x2 - calculatedX2).^2, 2) < thresh);
    if (ConsensusCount >= maxConsensusCount)
        maxConsensusCount = ConsensusCount;
        H = tempH;
    end
end
calculatedX2 = x1*H';
calculatedX2 = calculatedX2./calculatedX2(:,3);
ConsensusSet = sum((x2 - calculatedX2).^2, 2) < thresh;
p1 = x1(ConsensusSet, :);
p2 = x2(ConsensusSet, :);
H = homography(p1, p2);
end