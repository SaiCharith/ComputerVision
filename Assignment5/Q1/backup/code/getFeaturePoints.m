function salientFeaturePoints = getFeaturePoints(I, NUM_OF_POINTS, TEMPLATE_SIZE)
    [height, width] = size(I);
    featurePoints = detectSURFFeatures(I(TEMPLATE_SIZE+1:height-TEMPLATE_SIZE,TEMPLATE_SIZE+1:width-TEMPLATE_SIZE));
    featurePoints.Location = featurePoints.Location + TEMPLATE_SIZE;
    [Ix, Iy] = imgradientxy(I);

    Ix2 = Ix.*Ix;
    Iy2 = Iy.*Iy;
    IxIy = Ix.*Iy;
    minEig = zeros(featurePoints.Count, 1);

    for i=1:featurePoints.Count
        point = round(featurePoints.Location(i,:));
        pointy = point(1);
        pointx = point(2);
        a = sum(sum(Ix2(pointx-TEMPLATE_SIZE:pointx+TEMPLATE_SIZE, pointy-TEMPLATE_SIZE:pointy+TEMPLATE_SIZE)));
        b = sum(sum(Iy2(pointx-TEMPLATE_SIZE:pointx+TEMPLATE_SIZE, pointy-TEMPLATE_SIZE:pointy+TEMPLATE_SIZE)));
        c = sum(sum(IxIy(pointx-TEMPLATE_SIZE:pointx+TEMPLATE_SIZE, pointy-TEMPLATE_SIZE:pointy+TEMPLATE_SIZE)));
        minEig(i) = min(abs(eig([a,c;c,b])));
    end

    [~, indices] = maxk(minEig, NUM_OF_POINTS);
    salientFeaturePoints = featurePoints(indices);
end