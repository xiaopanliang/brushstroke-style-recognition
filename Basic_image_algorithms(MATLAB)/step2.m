function step2(thread_tmp_dir)
    load(strcat(thread_tmp_dir, 'step1_img'));
    heiheiim = zeros(size(im2,1),size(im2,2));
    for i = 1:size(im2,1)
        for j = 1:size(im2,2)
            if im2(i,j)==0
                heiheiim(i,j)= 1;
            else
            heiheiim(i,j)= 0; 
            end
        end
    end
    % 
    % %can change the 4 to 8, which means 8-connected 
    BW = bwlabel(heiheiim,4);
    index = 1;
    Region_collection = [];
    Real_brush_collection = [];
    for i = 1: max(max(BW))
        [r,c] = find(BW==i);
        Region_collection{i} = [r,c];
        %can change the maximum and the minimum
        if size(Region_collection{i},1)<=200 && size(Region_collection{i},1)>=0
        Real_brush_collection{index} = Region_collection{i};
        index = index + 1;
        end
    end
    re_im = zeros(size(im2,1),size(im2,2));
    for i = 1:size(Real_brush_collection,2)
        for j = 1:size(Real_brush_collection{i},1)
            re_im(Real_brush_collection{i}(j,1),Real_brush_collection{i}(j,2))= 1;
        end
    end
    severely_branch=[];
    theedgelist = cell(1,size(Real_brush_collection,2));
    junction = cell(1,size(Real_brush_collection,2));
    for i = 1:size(Real_brush_collection,2)
        brush = zeros(size(im2,1),size(im2,2));
        for j = 1:size(Real_brush_collection{i},1)
            brush(Real_brush_collection{i}(j,1),Real_brush_collection{i}(j,2))=1;
        end
        try
        [edgelist3,~,~] = edgelink(brush,2,1);
        theedgelist{i} = edgelist3;
        skeleton_im = edgelist2image(edgelist3);
        [rj, cj,~,~]=findendsjunctions(skeleton_im,0);
        junction{i} = [rj,cj];
        catch
            theedgelist{i} = [];
            junction{i} = [];
        end
    end
    save(strcat(thread_tmp_dir, 'step2_junction.mat'),'junction');
    save(strcat(thread_tmp_dir, 'Real_brush_collection.mat'),'Real_brush_collection');
    % save('../neural-style-tf-master/small/step2_junction.mat','junction');
    save(strcat(thread_tmp_dir, 'step2_theedgelist.mat'),'theedgelist')
    % save('../neural-style-tf-master/small/step2_theedgelist.mat','theedgelist')
end