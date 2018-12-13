function after_first_judge(result_name, thread_tmp_dir)
    load(strcat(thread_tmp_dir, 'step1_img'), 'im2');
    load(strcat(thread_tmp_dir, 'step3_result.mat'), 'result');
    load(strcat(thread_tmp_dir, 'step3_severely_branch.mat'),'severely_branch');
    load(strcat(thread_tmp_dir, 'step2_theedgelist.mat'),'theedgelist');
    load(strcat(thread_tmp_dir, 'Real_brush_collection.mat'),'Real_brush_collection');
    index = 1;
    % size(Real_brush_collection,2)
    for i=1:size(theedgelist,2)
        if severely_branch(i) == 0
            re_image = zeros(size(im2,1),size(im2,2));
            for ii = 1:size(result{index},1)
                re_image(result{index}(ii,1),result{index}(ii,2))=1;
            end
            index = index + 1;
            verynewedgelist = bwlabel(re_image,8);
            if max(max(verynewedgelist))>= 2
                severely_branch(i) = 1;
            end
        end
        if severely_branch(i) == 0
            brush_image = zeros(size(im2,1),size(im2,2));
            for ii = 1:size(Real_brush_collection{i},1)
                brush_image(Real_brush_collection{i}(ii,1),Real_brush_collection{i}(ii,2))=1;
            end
            [lbl,N] = bwlabel(brush_image);
            tol = 1;
            cntr = lbl == 1;
            d = bwdist(~cntr);
            max_dist = max(d(:));
            dists = d(abs(d - max_dist) <= tol);
            thickness = 2*mean(dists);
            thelength = sum(sum(re_image));
            broadness = thickness/thelength;
            if broadness <= 0.5 && broadness >=0.025
                severely_branch(i) = 0;
            else
                severely_branch(i) = 1;
            end
        end
        if severely_branch(i) == 0
            if sum(sum(brush_image))/2/sum(sum(re_image))/max_dist <= 0.5 || sum(sum(brush_image))/2/sum(sum(re_image))/max_dist >= 2
                severely_branch(i) = 1;
            end
        end
    end
    test = zeros(size(im2,1),size(im2,2));
    for i = 1:size(Real_brush_collection,2)
        if severely_branch(i)==0
            for j = 1:size(Real_brush_collection{i},1)
                test(Real_brush_collection{i}(j,1),Real_brush_collection{i}(j,2))= 1;
            end
        end
    end
    imwrite(test,result_name)
end