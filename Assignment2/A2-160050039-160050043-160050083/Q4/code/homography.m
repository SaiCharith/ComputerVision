function [ H ] = homography( p1, p2 )
%% p2 = H*p1 where p1 & p2 are correspondences and are in homogeneous coordinates

M = zeros(2*size(p1, 1), 9);
% Calculating matrix M
for i = 1:2*size(p1,1)
% Setting rows of M
    if (mod(i,2) == 1) % Odd Rows
        p = -p1((i+1)/2, :);
        px = -p*p2((i+1)/2, 1);
        z = [0, 0, 0];
        M(i, :) = [p, z, px];
    else % Even Rows
        p = -p1(i/2, :);
        py = -p*p2(i/2, 2);
        z = [0, 0, 0];
        M(i, :) = [z, p, py];
    end
end

[~, ~, V] = svd(M);
h = V(:, end); % The last column of matrix V
h = h./norm(h);
H = reshape(h, 3, 3)';
end