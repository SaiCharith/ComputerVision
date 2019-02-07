function res = Rot_Trans(Img,theta,xt,yt) 
    res = imrotate(Img,theta,'bilinear','crop');
    res = imtranslate(res,[xt,yt]);

end