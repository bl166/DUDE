% This script is for some post processing 
% genrated figures 8 and 9


I_tru = mat2gray(imread('./source_images/lena_halftone.png'));
I_cor = mat2gray(imread('./corrupted_images/corrupted_im_0_delta_0.02.png'));
I_res = mat2gray(imread('./result_figures/uu/corrected_im_size_0_order_14_delta_0.02_err_0.0133571428571.png'));
%%
err = (I_res - I_tru)~=0;
figure, imshow(err)
imwrite(mat2gray(err),['err_map_pe',num2str(nnz(err)/numel(I_tru)),'.png'])

overlay = I_tru;
overlay(err) = 0;
subs = zeros([size(I_tru),3]);
subs(:,:,1) = I_tru;
subs(:,:,2) = overlay;
subs(:,:,3) = overlay;
figure, imshow(subs)
imwrite(mat2gray(subs),['err_overlay_pe',num2str(nnz(err)/numel(I_tru)),'.png'])

nnz(err)/numel(I_tru)

%% other filters
figure
I_med = medfilt2(I_cor,[3 3],'symmetric');
nnz(I_med - I_tru)/numel(I_tru)
subplot(121),imshow(I_med)

I_wie = bwmorph(I_cor,'close');
I_wie(1,:)=1;I_wie(:,1)=1;I_wie(end,:)=1;I_wie(:,end)=1;
nnz(I_wie - I_tru)/numel(I_tru)
subplot(122), imshow(I_wie)

%% resize image
I_tru_resize = imresize(I_tru,[350,350],'nearest')~=0;
imwrite(I_tru_resize,'test_350.png')

%% roc
[tpr,fpr,thresholds] = roc(I_tru(:),I_res(:));
plotroc(I_tru(:),I_res(:));