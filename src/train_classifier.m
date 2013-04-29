function output = classify_image(filepath)
    [SVMstr, C] = train_classifier();
    
    % Extract sift features from image
    I = single(rgb2gray(im2double(imread(filepath)))) ;
    [~, descriptors] = vl_sift(I, 'PeakThresh', 0.01);
    img_hist = zeros(1, num_centers);
    ncols = size(descriptors, 2);
    clusters = size(C, 2);
    for col=1:ncols
        dist = inf;
        cluster_idx = -1;
        for clus=1:clusters
            tmp_dist = sqrt(sum((diff([descriptors(:,col); C(:,clus)])).^2));
            if(tmp_dist < dist)
                dist = tmp_dist;
                cluster_idx = clus;
            end
        end
        img_hist(1, cluster_idx) = img_hist(1, cluster_idx) + 1;
    end
    % Classify image
    classification = svmclassify(SVMstr, img_hist);
    output = classification;
end

function [output, centers] = train_classifier()
    directory = 'images\*.jpg';
    files = dir(directory);
    
    % Extracting features using SIFT
    [sift_vectors sift_vectors_cluster] = run_sift(files);
    
    % Running K Means Clustering
    ncols = size(sift_vectors_cluster, 2);
    %k = ncols / length(files);
    k = 10;
    sift_vectors_cluster = single(sift_vectors_cluster);
    [C, idx] = kmeans(sift_vectors_cluster, k);
    %k = size(C, 2);
    
    % Generating Image Histogram in terms of K centers
    img_hist = generate_image_histogram(files, sift_vectors, idx, k);
    
    % Train SVM
    group = ones(length(files));
    SVMstruct = svmtrain(img_hist, group, 'Kernel_Function', 'rbf');
    output = SVMstruct;
    centers = C;
end

function [sift_vectors, sift_vectors_cluster] = run_sift(image_files)
    sift_vectors = cell(1, length(image_files));    
    sift_vectors_cluster = [];
    for i=1:length(image_files)
        filepath = strcat('circle\', image_files(i).name)
        I = im2double(imread(filepath));
        I = single(rgb2gray(I)) ;
        
        % Extract Features
        [frames, descriptors] = vl_sift(I, 'PeakThresh', 0.01);
        
        sift_vectors{i} = descriptors;
        sift_vectors_cluster = [sift_vectors_cluster descriptors] ;
    end
end

function img_hist = generate_image_histogram(image_files, sift_vectors, idx, num_centers)
    img_hist = zeros(length(image_files), num_centers);
    start_index = 0;
    count = 1;
    for sift_vector=sift_vectors
        ncols = size(sift_vector{1}, 2);
        for col=1:ncols
            clust = idx(start_index+col);
            img_hist(count, clust) = img_hist(count, clust) + 1;
        end
        start_index = start_index + ncols;
        count = count + 1;
    end
end
