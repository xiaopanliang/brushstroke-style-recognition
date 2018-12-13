I = imread('smallonion.jpg');
new_image=[];
red_channel = I(:,:,1);
green_channel = I(:,:,2);
blue_channel = I(:,:,3);
grayI = rgb2gray(I);
[Gx, Gy] = imgradientxy(grayI);
new_image(:,1) = red_channel(:);
new_image(:,2) = green_channel(:);
new_image(:,3) = blue_channel(:);
new_image(:,4) = Gx(:);
new_image(:,5) = Gy(:);
missing = [];
fullindex = 0;
old_image = new_image;
index = 1;
% c2 = kmeans(c(:),2,'MaxIter',10000000,'Replicates',2);
% p2 = reshape(c2, size(red_channel));
% tkmeans2 = mean(c2);
for nihao = 1:15
    c = kmeans(new_image,2,'MaxIter',10000000);
    new_image = old_image;
    for j = 1:1:length(missing)
        c = [c(1:missing(j)-1);0;c(missing(j):end)];
    end
    tkmeans = mean(c);
    disp(tkmeans)
    c2 = reshape(c, size(grayI));
    for i = 1:size(c2,1)
        for  j = 1:size(c2,2)
            if c2(i,j) == 2
                grayI(i,j) = 255;
            else
                grayI(i,j) = 0;
            end
        end
    end
    BW = bwlabel(grayI,4);
    Region_collection = [];
    for i = 1: max(max(BW))
        [r,c] = find(BW==i);
        Region_collection = [r,c];
        if size(Region_collection,1)<=400 && size(Region_collection,1)>=1
            for ii = 1:length(r)
                missing(index) = (c(ii)-1)*size(Region_collection,1)+r(ii);
                index = index +1;
            end
        end
    end
    figure
    imshow(grayI);
    missing = sort(unique(missing));
    fullindex  = length(missing);
    index = fullindex + 1;
    for i = length(missing):-1:1
        new_image(missing(i),:)=[];
    end
%     fullindex  = length(missing);
%     missing_index = length(missing);
%     new_image = zeros(size(old_image,1)-length(missing),5);
%     for i = length(new_image):-1:1
%         if missing_index ~= 0
%             if i~=missing(missing_index)
%                 new_image(i,:)=old_image(i,:);
%             else
%                 new_image(i,:)=old_image(i-1,:);
%                 missing_index =  missing_index-1;
%             end
%         else
%             new_image(i,:)=old_image(i,:);
%         end
%     end
%     for i = length(missing):-1:1
%         new_image(missing(i),:) = [];
%     end
end