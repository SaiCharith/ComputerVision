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

    myassert(size(coords),size(zeros(3, num_points)));
    
    dwdp = zeros(num_points, 2, 6);
    for i=1:num_points
        dwdp(i,:,:) = [coords(1, i), coords(2, i), 1, 0, 0, 0; 0, 0, 0, coords(1, i), coords(2, i), 1];
    end
    
    myassert(dwdp, zeros(num_points, 2, 6));
    
    template = getPatch(frame1, coords);
    myassert(size(template), [num_points, 1]);
    
    p = [1 0 0; 0 1 0];  % Initial Guess
    L2error = 100000;
    t = 0;
    while L2error > 1*TEMPLATE_SIZE && t < 1000
        t = t+1;
        H = zeros(6,6);
        di = [getPatch(Ix, p*coords); getPatch(Iy, p*coords)];
        for i=1:num_points
            didwdp = di(:,i)'*squeeze(dwdp(i,:,:));
            myassert(size(didwdp), [1,6]);
            Htemp = didwdp'*didwdp;
            myassert(size(Htemp), [6,6]);
            H = H + Htemp;
        end
        Hinv = pinv(H);
        ptemp = zeros(1,6);
        for i=1:num_points
            idiff = template(i) - getPatch(frame2, p*coords(:,i));
            didwdp = di(:,i)'*squeeze(dwdp(i,:,:));
            ptemp = ptemp + idiff*didwdp;
        end
        myassert(size(ptemp), [1,6]);
        dp = ptemp*Hinv;
        p = p + reshape(dp, 3, 2)';
        L2error = sqrt(sum((template-getPatch(frame2, p*coords)).^2));
    end
    motion = p;
    L2error = sqrt(sum((template-getPatch(frame2, motion*coords)).^2));
end

function myassert = myassert(a, b)
    myassert = size(a,1) == size(b, 1);
    myassert = and(myassert, size(a, 2) == size(b, 2));
end

function patch = getPatch(I, coords)
    ind = sub2ind(size(I), max(min(round(coords(1,:)), size(I,1)), 1), max(min(round(coords(2,:)), size(I,2)), 1));
    patch = I(ind);
end