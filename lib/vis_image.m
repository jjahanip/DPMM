function [ ax ] = vis_image(dataset_name,  biomarker_name )
%VIS_IMAGE visualizes the selected biomarker
%   dataset_name = name of the dataset
%   biomarker_name = name of the biomarker
%   h = handle to the axes of image


% get the image name and color
ch = get_channels(dataset_name);
which_biomarker = cell2mat(cellfun(@(x) strcmp(x, biomarker_name), ch, 'un', 0));
image_name = ch{which_biomarker, 2};
image_color = ch{which_biomarker, 3};

% create the original image and color iamge
image =  imadjust(imread(image_name));
color_image = cat(3, ...
    image_color(1) * ones(size(image)),...
    image_color(2) * ones(size(image)),...
    image_color(3) * ones(size(image)));

% show the image
figure, 
ax = gca;
imshow(zeros(size(image))); hold on;
h = imshow(color_image);                                                    % show the color image
set (h, 'AlphaData',image)                                                  % set the transparecy of the color image to the image of the channel

end

