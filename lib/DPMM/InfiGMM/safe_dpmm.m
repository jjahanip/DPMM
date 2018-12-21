function [clusters] = safe_dpmm( features, T, iter)
%SAFE_DPMM run non-parametric clustering to generate cluster labels
%
% INPUTS:
% features = N*d feature matrix of N samples with d features
% T = maximum number of clusters
% iter = number of iterations to run dpmm

ids = (1:size(features))';

% keep a copy of original features
org_features = features;
org_id = (1:size(org_features))';

N = size(features, 1);
Phi = 1/T*ones(N,T);
alpha = rand*10;
% Run EM!
all_removed_samples = [];
reverseStr = '';
for i = 1:iter
    [gamma,mu_0,lambda,W,nu] = Mstep(features,Phi,alpha);
    [Phi,alpha] = Estep(features,gamma,mu_0,lambda,W,nu);
    
    % some samples become nan -> temporary remove the samples
    t = any(isnan(Phi),2);       % find the samples with NaN
    
    % store removed samples to add them at the end
    removed_samples = org_id(ismember(org_features, features(t,:), 'row'));
    all_removed_samples = [all_removed_samples; removed_samples];
    
    Phi = Phi(~t, :);            % remove element from phi
    features = features(~t, :);  % remove element from the samples
    ids = ids(~t, :);            % remove element from the ids
    
    % Display the progress
    percentDone = 100 * i / iter;
    msg = sprintf('Percent done: %3.1f', percentDone); %Don't forget this semicolon
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
end
fprintf('\n');

% extract clusters
N = size(features, 1);
[~, Z_hat] = max(Phi,[],2);
Z = zeros(1,N);
UZ = sort(unique(Z_hat));
for i = 1:length(UZ)
    Z(Z_hat==UZ(i)) = i;
end
clusters = Z';
num_clustest = max(clusters);

% add removed samples as new cluster
clusters = [clusters; ones(length(all_removed_samples), 1) * num_clustest+1];
features = [features; org_features(all_removed_samples, :)];
ids = [ids; all_removed_samples];

% re-order to original order
[~, idx] = sort(ids);
features = features(idx, :);
clusters = clusters(idx, :);

assert(isequal(features, org_features),'features do not match.')

end

