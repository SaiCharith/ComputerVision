tic;
%% Hill
imageName1 = 'hill/1.JPG';
imageName2 = 'hill/2.JPG';
imageName3 = 'hill/3.JPG';
threshold = 1;
superImage = myScript(imageName1, imageName2, imageName3, threshold);
figure; imshow(superImage);
%% Ledge
imageName1 = 'ledge/1.JPG';
imageName2 = 'ledge/2.JPG';
imageName3 = 'ledge/3.JPG';
threshold = 1;
superImage = myScript(imageName1, imageName2, imageName3, threshold);
figure; imshow(superImage);
%% Monument
imageName1 = 'monument/1.JPG';
imageName2 = 'monument/2.JPG';
imageName3 = 'monument/2.JPG';
threshold = 1;
superImage = myScript(imageName1, imageName2, imageName3, threshold);
figure; imshow(superImage);
%% Pier
imageName1 = 'pier/1.JPG';1
imageName2 = 'pier/2.JPG';
imageName3 = 'pier/3.JPG';
threshold = 1;
superImage = myScript(imageName1, imageName2, imageName3, threshold);
figure; imshow(superImage);
%% Sunset
imageName1 = 'sunset/1.JPG';
imageName2 = 'sunset/2.JPG';
imageName3 = 'sunset/3.JPG';
threshold = 1;
superImage = myScript(imageName1, imageName2, imageName3, threshold);
figure; imshow(superImage);
toc;