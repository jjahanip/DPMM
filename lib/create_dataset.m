function [ dataset ] = create_dataset( db, type, opt )
%CONCATENATE functions concatenate the db of different scales to create a
%dataset with two field of "features" and "labels" to be classified using
%any classification algorithm
fprintf('concatenating the databases of different scales...\n')
if nargin < 3
    opt.save = 'off';
end

for i = 1:length(db)-1
    if not(db{i}.dataset == db{i+1}.dataset )
        fprintf('datasets are not same !')
        keyboard;
    end
end
        

if strcmp (opt.save,'on')
    ds = {'LiVPa' , 'sham' , 'vehicle'};
    path = fileparts(db{1}.src.files{1});
    spltNm = strsplit(path, '/');
    if isempty ( strfind(path , 'multi') )
        saveFn = strcat( ds{ str2num(spltNm{end-3}) }, '_', ...
            spltNm{end-2}, '_', ...
            num2str(length(db)), '_', 'dataset' );
        savePath = strcat (cd(), '/results/', ds{ str2num(spltNm{end-3}) }, '/', ...
                           spltNm{end-2});
    else
        saveFn = strcat( ds{ str2num(spltNm{end-4}) }, '_', ...
            spltNm{end-3}, '_', ...
            spltNm{end-2}, '_', ...
            num2str(length(db)), '_', 'dataset' );
        savePath = strcat (cd(), '/results/', ds{ str2num(spltNm{end-4}) }, '/', ...
                           spltNm{end-3});
    end
    fN = strcat(savePath, '/', saveFn);
end

if length(db) == 1
    fprintf('This is a single scale dataset. No need for concatenation. \n ')
    dataset.features = db{1}.features;                                      % create the feature matrix
    dataset.labels = [db{1}.src.objects(:).class];                          % create the label matrix
    dataset.scales = length(db);                                            % store the number of scales in a variable
end

% and compare the equality of the datasets
for i = 1:length(db)
    % extract the centers of the datasets
    [~,file_names] = cellfun(@fileparts , db{i}.src.files , 'un' , 0);      % extract the file name
    centers_cell   = cellfun(@(x) strsplit(x, '_') , file_names , 'un' , 0);% split the file name to get the centers
    centers{1,i}(:,1)   = cell2mat(cellfun(@(x) str2num(x{1}) , centers_cell , 'un' , 0));     % store the x coordinate in the centers cell
    centers{1,i}(:,2)   = cell2mat(cellfun(@(x) str2num(x{2}) , centers_cell , 'un' , 0));     % store the y coordinate in the centers cell
    centers{1,i} = unique(centers{1,i}, 'rows', 'stable');                  % remove duplicated rows and keep the order
    % sort the centers based on x coordinate, if tie go to y coordinate
    [centers{1,i}, centers{2,i}]  = sortrows(centers{1,i},[1 2]);           % and save the indices of sorted centers in second row of cell
    
    % find and correct the positive and negative cells in the dataset
    negatives = [db{i}.src.objects(:).class] == 1;
    positives = [db{i}.src.objects(:).class] == 2;
    [db{i}.src.objects(negatives).class] = deal(0);
    [db{i}.src.objects(positives).class] = deal(i);
            
    % check the equality of the samples (centers) in each dataset
    if i == 1
        dataset.features = db{1}.features(:,centers{2,i});                  % create the sorted feature matrix
        dataset.labels = [db{1}.src.objects(centers{2,i}).class];           % create the sorted label matrix
        
    else
        if ~isequal(centers{1,i} , centers{1,i-1})
            fprintf('databases have different samples!!!\n')
            keyboard;
        end
        if strcmp(type , 'horz')
            dataset.features = horzcat(dataset.features, db{i}.features(:,centers{2,i})); 
            dataset.labels = horzcat(dataset.labels,[db{i}.src.objects(centers{2,i}).class]);
            dataset.concatenate = 'horz';
        elseif strcmp(type , 'vert')
            dataset.features = vertcat(dataset.features, db{i}.features(:,centers{2,i}));
            dataset.labels = vertcat(dataset.labels,[db{i}.src.objects(centers{2,i}).class]);
            dataset.concatenate = 'vert';
        end
    end
end
dataset.scales = length(db);                                                % store the number of scales in a variable
dataset.centers = centers{1,i};                                             % store the centers of the dataset
% transpose the matrices for this project
dataset.features = dataset.features';
dataset.labels = dataset.labels';
dataset.name = db{1}.dataset;

% save dataset
if strcmp (opt.save,'on')
    save(fN, 'dataset' , '-v7.3');
end

fprintf('done!\n');
end

