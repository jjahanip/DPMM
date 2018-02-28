function setPath()
% setPath()
%
% Update the search path with all the directories
% necessary for running the algorithm.
%
% 

basepath = cd();

images_path = 'D:\Jahandar\Lab\images';
input_path = 'D:\Jahandar\Lab\research\codes\me\hierachical_clustering\input';


addpath(genpath([basepath, '\lib']));
addpath(genpath([images_path , '\crops_for_badri_proposal']));
addpath(genpath([images_path , '\xiaoyang']));
addpath(genpath(input_path));



