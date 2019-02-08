function [ H ] = homography( p1, p2 )
%HOMOGRAPHY Summary of this function goes here
% World Coordinate = X = p1
% Image Coordinate = x = p2
% p2 = H*p1
%   Detailed explanation goes here

    M = zeros(8, 9);
    
    for i = 1:8
    % Setting rows of M
        if (mod(i,2) == 1)
            p = -[p1((i+1)/2, :), 1];
            px = -p*p2((i+1)/2, 1);
            z = [0, 0, 0];
            M(i, :) = [p, z, px];
        else
            p = -[p1(i/2, :), 1];
            py = -p*p2(i/2, 2);
            z = [0, 0, 0];
            M(i, :) = [z, p, py];
        end
    end
    
    [U, S, V] = svd(M);
    h = V(:, 9);
    H = reshape(h, 3, 3)';
end

