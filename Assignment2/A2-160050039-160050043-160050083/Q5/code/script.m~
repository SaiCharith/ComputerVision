
%%
tic;
I = imread('../input/barbara.png');
f1 = I;
I = imread('../input/flash1.jpg');
f2 = rgb2gray(I);
I = imread('../input/negative_barbara.png');
m1 = I ;
I = imread('../input/noflash1.jpg');
m2 = rgb2gray(I);

m1 = Rot_Trans(m1,23.5,-3,0);
m2 = Rot_Trans(m2,23.5,-3,0);

m1 = m1 + 8*floor(random('Uniform',0,1));
m1(m1>255) = 255;
m1(m1<0) = 0;
m2 = m2 + 8*floor(random('Uniform',0,1));
m2(m2>255) = 255;
m2(m2<0) = 0;

theta_range = -60:1:60 ;
x_range = -12:1:12 ;
s1 = double(zeros(121,25));
s2 = double(zeros(121,25));


I1=1;
J1=1;
I2=1;
J2=1;

mx1 = inf;
mx2 = inf;

for i = 1:size(theta_range,2)
    tic;
    for j = 1:size(x_range,2)
        Img1 = Rot_Trans(f1,theta_range(i),x_range(j),0);        
        s1(i,j) = jointentropy(m1,Img1);
        if s1(i,j)<mx1
            I1=i;
            J1=j;
            mx1 = s1(i,j);
        end
            
        Img2 = Rot_Trans(f2,theta_range(i),x_range(j),0);
        s2(i,j) = jointentropy(m2,Img2);
        
        if s2(i,j)<mx2
            I2=i;
            J2=j;
            mx2=s2(i,j);
        end       
    end
    toc;
end

display('barbara');
display(strcat('angle:',num2str(theta_range(I1))));
display(strcat('x:',num2str(x_range(J1))));
display('flash');
display(strcat('angle:',num2str(theta_range(I2))));
display(strcat('x:',num2str(x_range(J2))));

surf(x_range,theta_range,s1);
title('barbara');
xlabel('x');
ylabel('Theta');
zlabel('Joint Entropy');
figure();
surf(x_range,theta_range,s2);

xlabel('x');
ylabel('Theta');
zlabel('Joint Entropy');
title('flash');

AI1 = Rot_Trans(m1,-theta_range(I1),-x_range(J1),0);
AI2 = Rot_Trans(m2,-theta_range(I2),-x_range(J2),0);
figure();
imshow(AI1);
title('Barbara Aligned');
figure();
imshow(AI2);
title('noFlash Aligned');


%%

m3 = Rot_Trans(f1,0,256,0);
s3 = double(zeros(121,25));


mx3=0;
for i = 1:size(theta_range,2)
    for j = 1:size(x_range,2)
        Img1 = Rot_Trans(f1,theta_range(i),x_range(j),0);        
        s3(i,j) = jointentropy(m3,Img1);
        if s1(i,j)<mx3
            I3=i;
            J3=j;
            mx3 = s3(i,j);
        end     
    end
end

figure();
surf(x_range,theta_range,s3);
toc;





