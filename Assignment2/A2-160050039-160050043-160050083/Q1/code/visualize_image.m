function visualize_image(filename)
    figure;
    disp(filename);
    I=imread(strcat('../input/',filename));
    imshow(I);
    impixelinfo;

end