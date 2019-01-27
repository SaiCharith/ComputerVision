function [result_pose, composed_rot] = transformPose(rotations, pose, kinematic_chain, root_location)
    % rotations: A 15 x 3 x 3 array for rotation matrices of 15 bones
    % pose: The base pose coordinates 16 x 3.
    % kinematic chain: A 15 x 2 array of joint ordering
    % root_positoin: the index of the root in pose vector.
    % Your code here 
    
    %step one homogeneous coordinates
    pose=[pose ones(size(pose,1),1)];
    % make rotation in 4d space
    rotations_4=zeros(size(rotations,1),4,4);
    for i=1:size(rotations,1)
        rotations_4(i,:,:)=[squeeze(rotations(i,:,:)) zeros(3,1);0,0,0,1];
    end
    [result_pose,composed_rot]=dfs(rotations_4,pose,kinematic_chain,root_location,eye(4));
    result_pose=result_pose(:,1:end-1);
end

function [newpos,composed_rot]=dfs(rotations,pose,kinematic_chain,root_location,matrix_upto_now)
    newpos=pose;
    for i=1:size(kinematic_chain,1)
        parent=kinematic_chain(i,2);
        matrix_upto_now_2=matrix_upto_now*[eye(3) newpos(parent,1:end-1)';0,0,0,1]*squeeze(rotations(i,:,:))*[eye(3) -newpos(parent,1:end-1)';0,0,0,1];
        if parent==root_location
            child=kinematic_chain(i,1);
            [newpos,composed_rot]=dfs(rotations,newpos,kinematic_chain,child,matrix_upto_now_2);
        end
    end
    
    newpos(root_location,:)=matrix_upto_now*pose(root_location,:)';
    composed_rot(parent,:,:)=matrix_upto_now;
end
