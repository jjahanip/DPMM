clc; clear;
setPath();


%% load data
load LiVPA_S100_multi_3.mat               ; new_db{1} = db{1};       
%     load LiVPA_APC_multi_3.mat          ; new_db{2} = db{1};
%     load LiVPA_GFAP_multi_3.mat         ; new_db{1} = db{3};
%     load LiVPA_GLAST_multi_3.mat        ; new_db{2} = db{3};
biomarkers = cellfun(@(S) S.biomarker, db, 'UniformOutput', false);

%% create dataset
type = 'vert';
dataset = create_dataset( new_db, type );

dataset.features = double(dataset.features);
dataset.features = dataset.features./max(dataset.features(:));              % normalize the deep features to [0,1]
%% plot original features 
% figure,
% for i=1:681
%     subplot(27,27,i)
%     hist(dataset.features(:,i), 200)
%     axis off
% end
% suptitle('histogram of original features')

%% PCA:
% PCA and project the data to 2D
num_feat = 100;
[U, S] = pca(dataset.features);
dataset.red_feat = projectData(dataset.features, U, num_feat);
% normalize [0-1]
new_feat = bsxfun(@rdivide, ... 
                  bsxfun(@minus, dataset.red_feat, min(dataset.red_feat)),...
                  max(dataset.red_feat) - min(dataset.red_feat)...
                  ); 
%% plot normalized features after PCA
% figure,
% num_plots = ceil(sqrt(num_feat));
% for i=1:num_feat
%     subplot(num_plots,num_plots,i)
%     hist(new_feat(:,i), 200)
% end
% title_text = sprintf('histogram of %d PCA-reduced features', num_feat);
% suptitle(title_text)
%% DPMM:
N = size(new_feat, 1);
T = 50;
Phi = 1/T*ones(N,T);
alpha = rand*10;
% Run EM!
for i = 1:100
    [gamma,mu_0,lambda,W,nu] = Mstep(new_feat,Phi,alpha);
    [Phi,alpha] = Estep(new_feat,gamma,mu_0,lambda,W,nu);
    t = any(isnan(Phi),2);    % find the samples with NaN
    Phi = Phi(~t, :);         % remove element from phi
    new_feat = new_feat(~t, :); % remove element from the samples
    dataset.centers = dataset.centers(~t, :);
end

N = size(new_feat, 1);      % update labels with removes samples
[~, Z_hat] = max(Phi,[],2);
Z = zeros(1,N);
UZ = sort(unique(Z_hat));
for i = 1:length(UZ)
    Z(Z_hat==UZ(i)) = i;
end
classes = Z';

num_cls = max(classes);
%% visualize distribution of each feature for different clusters:
% for i = 1:max(classes)
%     clusters{i} = new_feat(classes == i, :);
% end
% 
% figure,
% for i=1:num_feat
%     for j=1:num_cls
%         subplot(num_feat, num_cls, num_cls*(i-1)+j)
%         hist(clusters{j}(:,i), 200)
%         axis off
%     end
% end
% suptitle('Histogram of different clusters for each feature');

%% calculate pdf:
index = linspace(0, 1, 1000);
for i = 1:num_feat
    myFit{i} = fitdist(new_feat(:,i), 'kernel', 'by', classes);
    myPdf{i} = cellfun(@(x) pdf(x,index), myFit{i}, 'uniformoutput', 0);
end

%% test calculated pdf (histogram, fitted pdf with hist and fitted pdf with kernel)
selected_feat = 2;
figure, hold on;
% hist(myFit{selected_feat}{1}.InputData.data, 1000);
histfit(myFit{selected_feat}{1}.InputData.data, 1000)
plot(index, pdf(myFit{selected_feat}{1}, index),'g', 'LineWidth',2);
legend({'histogram', 'fitted pdf based on histogram', 'fitted pdf using pdf function'})

%% visualize distribution of each cluster for different features:
% selected_cluster = 12;
% figure, hold on; count = 0; 
% for i=1:20:num_feat
%     visFit = fitdist(new_feat(classes == selected_cluster,i), 'kernel');
%     plot(index, pdf(visFit, index))
%     count = count+1;
% end
% suptitle(sprintf('PDFs of %d features for cluster # %d',count, selected_cluster))

%% visualize pdf of clusters of the all features
% figure,
% num_plots = ceil(sqrt(num_feat));
% for i=1:num_feat
%     visFit = fitdist(new_feat(:,i), 'kernel', 'by', classes);
%     subplot(num_plots,num_plots,i), hold on;
%     cellfun(@(x) plot(index, pdf(x, index)), visFit, 'uniformoutput', 0);
% end
% suptitle(sprintf('PDFs of %d clusters of the each feature',num_cls))

%% calculate distance between clusters
for i = 1:13
single_feat_pdf = cellfun(@(x) x{i}, myPdf, 'un', 0);
joint_pdf{i} = prod(cat(1, single_feat_pdf{:}), 1);
end

dist = KLDiv(pdf(myFit_1, index), pdf(myFit_2, index));
%% visualization
% read and show the image
image =  imadjust(imread('ARBc_#4_Li+VPA_37C_4110_C6_IlluminationCorrected_stitched.tif'));
color =  [1 1 0];
color_image = cat(3, ...
                       color(1) * ones(size(image)),...
                       color(2) * ones(size(image)),...
                       color(3) * ones(size(image)));
imshow(zeros(size(image))); hold on;                   
h = imshow(color_image);                                                    % show the color image
set (h, 'AlphaData',image)                                                  % set the transparecy of the color image to the image of the channel
% image(:,:,1) = imadjust(imread('ARBc_#4_Li+VPA_37C_4110_C6_IlluminationCorrected_stitched.tif'));
% image(:,:,2) = imadjust(imread('ARBc_#4_Li+VPA_37C_4111_C8_IlluminationCorrected_stitched_registered.tif'));
% image(:,:,3) = zeros(size(image(:,:,1)));
% imshow(imread('Composite_S100_GFAP_DAPI_GLAST.jpg'));
hold on;

% setting colormap information for cluster numbers
% cl_map_clust = [ 0 1 1;
%                 .6 0 .6;
%                  1 1 1; 
%                  1 1 0;
%                ];                                          % create a color map based on different clusters
% rng(0);                                                                     % set the seed for the random
% cl_map_clust = cl_map_clust(randperm(size(cl_map_clust,1)),:);              % shuffle the colors to have random colors

cl_map_clust = hsv (max(classes));                                          % create a color map based on different clusters
for i = 1:max(classes)
    plot( dataset.centers(classes == i,1), ...
          dataset.centers(classes == i,2), ...
          '.', ...
          'color', cl_map_clust(i,:), ...
          'markersize',25 ...
        );
end



%% T-SNE:
tsne_labels = cl_map_clust(classes,:) ;
new_feats = tsne(dataset.features, tsne_labels);
set(gca,'Color',[0 0 0])

figure, 
for i = 1:max(classes)
    plot( dataset.centers(classes == i,1), ...
          dataset.centers(classes == i,2), ...
          '.', ...
          'color', cl_map_clust(i,:), ...
          'markersize',25 ...
        );
end

%% add scalebar:
bar = 200;

pixel = .325;
width = bar / pixel;

p1 = [8000 8000+width];
p2 = [600 600];
plot(p1, p2, 'y-', 'LineWidth', 2);
str = strcat(int2str(bar) , ' \mum');
h = text(8000, 400, str, 'color', 'y', 'FontSize', 25);

