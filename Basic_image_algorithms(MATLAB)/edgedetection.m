function edgedetection(image, thread_tmp_dir)
    x = imread(image);
    x = rgb2gray(x);
    y = edge(x,'Canny',[0.02,0.04]);
    [edgelist, ~] = edgelink(y,10,0);
    edgelist = {size(x,1),size(x,2),edgelist};
    save(strcat(thread_tmp_dir, 'step1_edgelist.mat'),'edgelist');
end