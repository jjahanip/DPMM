function [  ] = plot_heatmap(dist,  num_samples )
%plot_heatmap plots the heatmap of the distance between clusters and the
%number of samples in each cluster
%   dist = n*n symmetric matrix containing distances between clusters
%   num_sample = n*1 column vector having number of samples in each cluster


% create figure
f = figure;
fig_pos = get(f,'position');
set(f,'position',[fig_pos(1:2) fig_pos(3)*2 fig_pos(4)])    % make figure wide

% plot heatmap on the left
h1 = subplot(121);
imagesc(dist)
set(h1, 'Xtick', 1:length(dist));
set(h1, 'Ytick', 1:length(dist));
colormap('jet'), colorbar;
title('differences between clusters')

% plot number of samples in right
t = uitable(f, 'ColumnName', {'number of samples'},...
        'Data', num_samples);
subplot(122),plot(3)
pos = get(subplot(122),'position');
delete(subplot(122))
set(t,'units','normalized')
set(t,'position',pos)
% Auto-resize:
jScroll = findjobj(t);
jTable  = jScroll.getViewport.getView;
jTable.setAutoResizeMode(jTable.AUTO_RESIZE_SUBSEQUENT_COLUMNS);
drawnow;

end

