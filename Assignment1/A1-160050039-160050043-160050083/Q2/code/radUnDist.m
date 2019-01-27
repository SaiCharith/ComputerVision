function imOut = radUnDist(imIn, k1, k2, nSteps)
    % Your code here
    [m, n] = size(imIn);
    [x, y] = meshgrid(1:n, 1:m);
    
    cx = m/2;
    cy = n/2;
    x_orig = x/cx - 1;
    y_orig = y/cy - 1;
    
    x = x/cx - 1;
    y = y/cy - 1;
    
    for k=1:nSteps    
        r2 = sqrt(x.^2 + y.^2);
        dr = 1 + k1*r2 + k2*r2.^2;   
        x =  x_orig./dr;
        y =  y_orig./dr;
    end
    
    x = x*cx + cx;
    y = y*cy + cy;
    
    imOut = interp2(imIn, x, y, 'cubic');
end