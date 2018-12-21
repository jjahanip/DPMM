clc; clear;
setPath();


%% load data
% load LiVPA_S100_multi_3.mat;     new_db{1} = db{1};
% load LiVPA_APC_multi_3.mat;      new_db{1} = db{1};
% load LiVPA_GFAP_multi_3.mat;     new_db{1} = db{3};
% load LiVPA_GLAST_multi_3.mat;    new_db{2} = db{3};
load LiVPA_NeuN_multi_3.mat;     new_db{1} = db{1}; 
biomarkers = cellfun(@(S) S.biomarker, new_db, 'UniformOutput', false);

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
%% DPMM-1:
% 
% N = size(new_feat, 1);
% T = 10;
% Phi = 1/T*ones(N,T);
% alpha = rand*10;
% % Run EM!
% removed_samples = [];
% for i = 1:100
%     [gamma,mu_0,lambda,W,nu] = Mstep(new_feat,Phi,alpha);
%     [Phi,alpha] = Estep(new_feat,gamma,mu_0,lambda,W,nu);
%     t = any(isnan(Phi),2);    % find the samples with NaN
%     Phi = Phi(~t, :);         % remove element from phi
%     new_feat = new_feat(~t, :); % remove element from the samples
%     removed_samples = [removed_samples; dataset.centers(t, :)];
%     dataset.centers = dataset.centers(~t, :);
%     dataset.labels =  dataset.labels(~t, :);
%     % check how many + samples in each cluster
%     [~, Z_hat] = max(Phi,[],2);
%     UZ = sort(unique(Z_hat));
%     Z = zeros(1,N);
%     fprintf('round %d\n', i)
%     for j = 1:length(UZ)
%         Z(Z_hat==UZ(j)) = j;
%         acc = sum(ismember(find(Z == j), find(dataset.labels))) / sum(dataset.labels);
%         len = length(find(Z == j));
%         fprintf('class %d:\t%d cells\t%.2f pos\n', j, len, acc)
%     end
%     fprintf('----------------------------------------\n')
%     
% end
% 
% N = size(new_feat, 1);      % update labels with removes samples
% [~, Z_hat] = max(Phi,[],2);
% Z = zeros(1,N);
% UZ = sort(unique(Z_hat));
% for i = 1:length(UZ)
%     Z(Z_hat==UZ(i)) = i;
% end
% classes = Z';
% 
% num_cls = max(classes);
% 
%% DPMM-2:
max_cluster = 10;
iteration = 100;
classes_2 = safe_dpmm(new_feat, max_cluster, iteration);
%% calculate KL-Div
mu = mu_0(UZ, :);
sigma = W(:,:, UZ);

dist = zeros(num_cls, num_cls);
for i = 1: num_cls
    num_samples(i, 1) = sum(classes == i);                                  % num of samples in each class
    percent_pos(i, 1) = sum(ismember(find(classes == i), find(dataset.labels))) ...  % percent of bioM+ in cluster
                / sum(dataset.labels); 
    for j = i+1:num_cls
        dist_1 = KLDiv_multinorm(mu(i,:), mu(j,:), sigma(:,:,i), sigma(:,:,j));
        dist_2 = KLDiv_multinorm(mu(j,:), mu(i,:), sigma(:,:,j), sigma(:,:,i));
        dist(i,j) = dist_1 + dist_2;
        dist(j,i) = dist_1 + dist_2;
    end
end

% plot heatmap and number of samples
plot_heatmap(dist, num_samples, percent_pos)

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
% index = linspace(0, 1, 1000);
% for i = 1:num_feat
%     myFit{i} = fitdist(new_feat(:,i), 'kernel', 'by', classes);
%     myPdf{i} = cellfun(@(x) pdf(x,index), myFit{i}, 'uniformoutput', 0);
% end

%% test calculated pdf (histogram, fitted pdf with hist and fitted pdf with kernel)
% selected_feat = 2;
% figure, hold on;
% % hist(myFit{selected_feat}{1}.InputData.data, 1000);
% histfit(myFit{selected_feat}{1}.InputData.data, 1000)
% plot(index, pdf(myFit{selected_feat}{1}, index),'g', 'LineWidth',2);
% legend({'histogram', 'fitted pdf based on histogram', 'fitted pdf using pdf function'})

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

%% calculate joint pdf
% for i = 1:13
%     single_feat_pdf = cellfun(@(x) x{i}, myPdf, 'un', 0);
%     joint_pdf{i} = prod(cat(1, single_feat_pdf{:}), 1);
% end

%% visualization
ax = vis_image(dataset.name, 'RECA1');

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
h_all = plot(dataset.centers(:, 1), dataset.centers(:, 2), 'r.', 'markersize', 10);
h0 = plot(dataset.centers(dataset.labels==1, 1), dataset.centers(dataset.labels==1, 2), 'r+', 'markersize', 15);
hr = plot(removed_samples(:,1), removed_samples(:,2), 'r.', 'markersize', 25);
h1 = plot(dataset.centers(classes==1,1), dataset.centers(classes==1,2), 'b.', 'markersize', 15);
h2 = plot(dataset.centers(classes==2,1), dataset.centers(classes==2,2), 'b.', 'markersize', 15);
h3 = plot(dataset.centers(classes==3,1), dataset.centers(classes==3,2), 'g.', 'markersize', 15);
h4 = plot(dataset.centers(classes==4,1), dataset.centers(classes==4,2), 'c.', 'markersize', 15);
h5 = plot(dataset.centers(classes==5,1), dataset.centers(classes==5,2), 'c.', 'markersize', 15);
h6 = plot(dataset.centers(classes==6,1), dataset.centers(classes==6,2), 'c.', 'markersize', 15);
h7 = plot(dataset.centers(classes==7,1), dataset.centers(classes==7,2), 'y.', 'markersize', 15);
h8 = plot(dataset.centers(classes==8,1), dataset.centers(classes==8,2), 'y.', 'markersize', 15);
h9 = plot(dataset.centers(classes==9,1), dataset.centers(classes==9,2), 'c.', 'markersize', 15);
h10 = plot(dataset.centers(classes==10,1), dataset.centers(classes==10,2), 'c.', 'markersize', 15);

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


