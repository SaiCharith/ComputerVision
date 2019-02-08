threeDUnit=38;%(in mm)
threeDoffset=8;%(in mm)

x_3D=threeDoffset+[0,0,0,3,4,6]*threeDUnit;
y_3D=[4,5,6,4,6,5]*threeDUnit;
z_3D=-threeDoffset-[5,4,5,0,0,0]*threeDUnit;


% visualize_image('q1.jpeg');
% visualize_image('q2.jpeg');
% visualize_image('q3.jpeg');

% %q1
% 
% x_2D=[499,428,376,462,335,408];
% y_2D=[236,259,235,614,658,752];
% 
% %q2
% 
x_2D=[574,505,474,389,261,283];
y_2D=[514,512,461,842,812,910];
% 
% %q3
% 
% x_2D=[503,432,383,464,337,402];
% y_2D=[429,460,421,825,847,916];

%% FOR TESTING

% x_3D_norm=x_3D;
% y_3D_norm=y_3D;
% z_3D_norm=z_3D;
% 
% x_2D_norm=x_2D;
% y_2D_norm=y_2D;


%% Step1


x_3D_cen= x_3D-(sum(x_3D)/6.0);
y_3D_cen= y_3D-(sum(y_3D)/6.0);
z_3D_cen= z_3D-(sum(z_3D)/6.0);

dist_3D_avg=sum(sqrt(x_3D_cen.^2+y_3D_cen.^2+z_3D_cen.^2))/6.0;
s_3D=sqrt(3)/dist_3D_avg;
x_3D_norm=sqrt(3)/dist_3D_avg*(x_3D_cen);
y_3D_norm=sqrt(3)/dist_3D_avg*(y_3D_cen);
z_3D_norm=sqrt(3)/dist_3D_avg*(z_3D_cen);

U=[s_3D,0,0,-s_3D*(sum(x_3D)/6.0);	0,s_3D,0,-s_3D*(sum(y_3D)/6.0);0,0,s_3D,-s_3D*(sum(z_3D)/6.0);	0,0,0,1];


x_2D_cen= x_2D-(sum(x_2D)/6.0);
y_2D_cen= y_2D-(sum(y_2D)/6.0);


dist_2D_avg=sum(sqrt(x_2D.^2+y_2D.^2))/6.0;
s_2D=sqrt(2)/dist_2D_avg;
x_2D_norm=s_2D*(x_2D_cen);
y_2D_norm=s_2D*(y_2D_cen);


T=[s_2D,0,-s_2D*(sum(x_2D)/6.0);	0,s_2D,-s_2D*(sum(y_2D)/6.0);	0,0,1];


%Step2 DLT

no_of_points=6;

M=zeros(no_of_points*2,12);


for i = 1: no_of_points
	%x cord
	M(2*i-1,:)=[-x_3D_norm(i),-y_3D_norm(i),-z_3D_norm(i),-1,0,0,0,0,x_2D_norm(i)*x_3D_norm(i),x_2D_norm(i)*y_3D_norm(i),x_2D_norm(i)*z_3D_norm(i),x_2D_norm(i)];

	%y cord
	M(2*i,:)=[0,0,0,0,-x_3D_norm(i),-y_3D_norm(i),-z_3D_norm(i),-1,y_2D_norm(i)*x_3D_norm(i),y_2D_norm(i)*y_3D_norm(i),y_2D_norm(i)*z_3D_norm(i),y_2D_norm(i)] ;   
end


[~,~,V1]=svd(M);
p=V1(:,2*no_of_points);

p=reshape(p,4,3);
p=p';
p=inv(T)*p*U;

%Step3
H_inf=p(1:3,1:3);
H_hat=p(:,4);
inv_H_inf=inv(H_inf);
proj_center=-inv_H_inf*H_hat
[inv_R,inv_K]=qr(inv_H_inf);
R=inv(inv_R);
K=inv(inv_K);
K_Hat_Hat=K/K(3,3);

my_neg_mat=[-1,0,0;0,-1,0;0,0,1];
K_Hat_Hat=K_Hat_Hat*my_neg_mat
R=my_neg_mat*R

new_p=K_Hat_Hat*R*[eye(3) -proj_center]


%Step 4
error=zeros(no_of_points,1);
for i = 1:no_of_points
	Predicted_coordinates=new_p*[x_3D(i),y_3D(i),z_3D(i),1]';
	Predicted_coordinates=Predicted_coordinates/Predicted_coordinates(3);
	e = [x_2D(i), y_2D(i), 1]' - Predicted_coordinates;
	error(i)=sqrt(sum(e.^2));
end
disp('errors in the respective points are')
disp(error)
disp('average MSE is :- ')
disp(sum(error)/6)

%% Observation
% The Observed values of average MSE for q1 is 0.3344 q2 is  0.1795 and q3 is 0.6752 
% which is considerably small compared to the actual value of the x and y
% coordinates.

% The reason for normalizing is not clear from the above example as the coordinates are well behaved. If i run it without normalizing
% and also commenting line 88 then also i get almost the same MSE


% It is a good Idea to Normalize because Then we work with the same order
% of magnitude of the points. this not only helps in controlling precision
% errors but also keeps the numbers small enough to be well in range for
% our computation.

% So normalization is essential not only for numerical stability, 
% but also for more accurate estimation in presence of noise and faster 
% solution (in case of iterative solver).

% The precise answer as we got from 
% https://dsp.stackexchange.com/questions/10887/why-normalize-the-data-set-before-applying-direct-linear-transform?fbclid=IwAR0gwxXtF9ire5mFa13hjVlCnn7wPyD3-FFtTSMZWdc3uT6pGFOAvpSSH5s
% is that We decrease the condition number of the Matrix M which is the
% 12*12 matrix we use in the above computation