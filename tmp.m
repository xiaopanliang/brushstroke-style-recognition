I = imread('ori_img.png');
subplot(1,2,1)
imshow(I)
title('Original Image');
[L, Centers] = imsegkmeans(I, 2);
B = labeloverlay(I, L);
subplot(1,2,2)
imshow(B)
title('Labeled Image')