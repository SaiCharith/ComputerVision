function s = jointentropy(Img1,Img2)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    t=10;
    E = double(zeros(1+ceil(256/t),1+ceil(256/t)));
    [M,N] = size(Img1);
    for i = 1:M
        for j = 1:N
            E(1+floor(Img1(i,j)/t),1+floor(Img2(i,j)/t))=1+E(1+floor(Img1(i,j)/t),1+floor(Img2(i,j)/t));
        end
    end
    
    E = E/(M*N);
    E(E==0) = 1;
    
    s = - sum(sum(E.*log2(E)));


end

