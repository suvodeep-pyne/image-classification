clear all

files = dir('images\*.jpg');
sift_vectors = cell(1, length(files));
sift_vectors_cluster = [];

for i=1:length(files)
    temp = strcat('images\', files(i).name);
    I = im2double(imread(temp));
    I = single(rgb2gray(I)) ;
    [f, d] = vl_sift(I, 'PeakThresh', 0.01);
    sift_vectors{i} = d;
    sift_vectors_cluster = [sift_vectors_cluster d] ;
end

[rows, cols] = size(sift_vectors_cluster);
k = cols/length(files);
sift_vectors_cluster = single(sift_vectors_cluster) ;
[idx, C] = kmeans(sift_vectors_cluster, k);