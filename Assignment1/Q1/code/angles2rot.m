function rot_matrix = angles2rot(angles_list)
    %% Your code here
    % angles_list: [theta1, theta(2), theta(3)] about the x,y and z axes,
    % respectively.
    n_parts=size(angles_list,1);
    rot_matrix=zeros(n_parts,3,3);
    for i=1:n_parts
        theta=angles_list(i,:);
        X=[1,0,0;0,cosd(theta(1)),-sind(theta(1));0,sind(theta(1)),cosd(theta(1))];
        Y=[cosd(theta(2)),0,sind(theta(2));0,1,0;-sind(theta(2)),0,cosd(theta(2))];
        Z=[cosd(theta(3)),-sind(theta(3)),0;sind(theta(3)),cosd(theta(3)),0;0,0,1];
        rot_matrix(i,:,:)=Z*Y*X;
    end
        
end




