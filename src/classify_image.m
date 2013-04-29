function output = classify_image(filepath, SVMstr, C)
    % Extract sift features from image
    I = single(rgb2gray(im2double(imread(filepath)))) ;
    [~, descriptors] = vl_sift(I, 'PeakThresh', 0.001);
    img_hist = zeros(1, size(C, 2));
    ncols = size(descriptors, 2);
    clusters = size(C, 2);
    for col=1:ncols
        dist = inf;
        cluster_idx = -1;
        descript = double(descriptors(:,col));
        for clus=1:clusters
            tmp_dist = sqrt(sum((descript(:) - C(:,clus)).^2));
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