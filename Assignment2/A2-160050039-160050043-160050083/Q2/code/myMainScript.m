%% Rigit Transform between 2 sets of 3D Points

%% Load Data
load('../input/Q1data.mat');

%% Your code here
H = homography(p2, p1);
a = H\[p3(1, :), 1]';
a = a./a(3);
b = H\[p3(2, :), 1]';
b = b./b(3);
c = H\[p3(3, :), 1]';
c = c./c(3);
width = norm(a-b)
length = norm(b-c)