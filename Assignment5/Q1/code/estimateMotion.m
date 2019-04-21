function [motion, L2error] = estimateMotion(frame1, frame2, point, TEMPLATE_SIZE, Ix, Iy)
    pointx = point(1);
    pointy = point(2);

    num_points = (2*TEMPLATE_SIZE+1)*(2*TEMPLATE_SIZE+1);
    coords = zeros(3, num_points);
    
    for i=1:(2*TEMPLATE_SIZE+1)
        coords(1,(i-1)*(2*TEMPLATE_SIZE+1)+1:i*(2*TEMPLATE_SIZE+1)) = pointx-TEMPLATE_SIZE+i-1;
        coords(2,(i-1)*(2*TEMPLATE_SIZE+1)+1:i*(2*TEMPLATE_SIZE+1)) = pointy-TEMPLATE_SIZE:pointy+TEMPLATE_SIZE;
        coords(3,(i-1)*(2*TEMPLATE_SIZE+1)+1:i*(2*TEMPLATE_SIZE+1)) = 1;
    end

    dwdp = zeros(num_points, 2, 6);     % 1e (v) Calculate the jacobian matrix
    for i=1:num_points
        dwdp(i,:,:) = [coords(1, i), coords(2, i), 1, 0, 0, 0; 0, 0, 0, coords(1, i), coords(2, i), 1];
    end
    
    template = getPatch(frame1, coords);    % 1c    Template patch
    p = [1 0 0; 0 1 0];  % 1e (i) Initial Guess for affine transformation
    L2error = 10000;
    t = 0;
    while L2error > 2*TEMPLATE_SIZE && t < 50
        t = t+1;
        dI = [getPatch(Ix, p*coords); getPatch(Iy, p*coords)];  %1e (iv)    Image gradients of warped patch
        dIdwdp = zeros(num_points, 6);
        H = zeros(6,6);
        for i=1:num_points
            dIdwdp(i, :) = reshape(dI(:,i), 1, 2)*squeeze(dwdp(i,:,:));
            H = H + dIdwdp(i, :)'*dIdwdp(i, :);     %1e (v) Compute hessian
        end
        warp = getPatch(frame2, p*coords);  % 1e (ii) Warped frame
        error = template-warp;
        temp = error*dIdwdp;
        delP = (temp*pinv(H))';
        p = p + reshape(delP, 3, 2)';
        L2error = sqrt(sum((template-warp).^2));    % 1e (iii)  L2 error
    end
    motion = p;
    L2error = sqrt(sum((template-getPatch(frame2, motion*coords)).^2));
end

function patch = getPatch(I, coords)
    if (coords(1,1) < 0 || coords(1,1) > size(I,1) || coords(2,1) < 0 || coords(2,1) > size(I,2))
        % Partial checking for Bad coords
        s = size(coords(1,:),2);
        patch = zeros(1,s);
    else
        ind = sub2ind(size(I), max(min(round(coords(1,:)), size(I,1)), 1), max(min(round(coords(2,:)), size(I,2)), 1));
        patch = I(ind);
    end
end