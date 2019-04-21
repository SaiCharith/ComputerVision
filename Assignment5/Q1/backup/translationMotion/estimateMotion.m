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
    
    template = getPatch(frame1, coords);
    p = [1 0 0; 0 1 0];  % Initial Guess
    L2error = 10000;
    t = 0;
    while L2error > TEMPLATE_SIZE && t < 40
        t = t+1;
        dI = [getPatch(Ix, p*coords); getPatch(Iy, p*coords)];
        Hinv = pinv(dI*dI');
        warp = getPatch(frame2, p*coords);
        delP = Hinv*(dI*(template-warp)');
%         p = [p(:,1:2)  ((p(:,3) + delP)/2)]   % Averaging
        p = [p(:,1:2)  ((p(:,3) + delP))];
        L2error = sqrt(sum((template-warp).^2));
    end
    motion = p
    L2error = sqrt(sum((template-getPatch(frame2, motion*coords)).^2));
end



function patch = getPatch(I, coords)
    if (coords(1,1) < 0 || coords(1,1) > size(I,1) || coords(2,1) < 0 || coords(2,1) > size(I,2))
        % Partial checking for Bad coords
        s = size(coords, 2);
        patch = zeros(1, s);
    else
        ind1 = sub2ind(size(I), max(min(floor(coords(1,:)), size(I,1)), 1), max(min(floor(coords(2,:)), size(I,2)), 1));
        ind2 = sub2ind(size(I), max(min(ceil(coords(1,:)), size(I,1)), 1), max(min(ceil(coords(2,:)), size(I,2)), 1));
        patch1 = I(ind1);
        patch2 = I(ind2);
        patch = (patch1 + patch2)/2;
    end
end