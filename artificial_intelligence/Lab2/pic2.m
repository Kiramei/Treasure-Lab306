
rebuild()

function [matrix, face_matrix] = get_data()
    % 读取图像数据，返回需要降维的矩阵和用于重建的样本矩阵。
    data_matrix = zeros(400, 56*46);
    for i = 1:40
        people_id = i;
        for j = 1:10
            img_path = sprintf('./Dataset/ORL/s%d/%d.pgm', people_id, j);
            image = imread(img_path);
            image = imresize(image,0.5);
            [rows, cols] = size(image);  % 图像像素
            img_vector = zeros(1, rows * cols);  % 全为0长度为rows * cols的向量
            img_vector = reshape(image, 1, rows * cols);  % 将img化成行向量
            data_matrix((i-1)*10+j, :) = img_vector;
        end
    end
    data_matrix = data_matrix';
    data_matrix = reshape(data_matrix, 56*46, 400);  % 转为二维数组
    datamatrix = data_matrix;
    data_matrix = data_matrix';
    matrix = data_matrix(1:300, :);
    face_matrix = data_matrix(201:250, :);
end

function [mean_matrix, select] = pca(matrix, k)
    % 主成分分析，返回均值矩阵和前k个特征向量组成的矩阵。
    matrix = single(matrix);
    mean_matrix = mean(matrix, 2);  % 按行计算平均
    matrix = matrix - mean_matrix;  % 去平均值
    cov_matrix = cov(matrix);  % 协方差矩阵
    [eigenvector, eigenvalue] = eig(cov_matrix);  % 特征值和特征向量
    eigenvalue = diag(eigenvalue);
    [~, index] = sort(eigenvalue, 'descend');  % 排序
    index = index(1:k);
    select = eigenvector(:, index);  % 最大特征值对应特征向量
end

function rebuild()
    % 重建样本矩阵，使用不同维度的特征向量进行重建，并显示重建后的图像。
    [matrix, face_matrix] = get_data();
    [mean_matrix, eigenvector] = pca(matrix, 160);
    for j = 1:5
        figure;
        subplot(3, 3, 1);
        face_matrix_ = uint8(face_matrix);
        imshow(reshape(face_matrix_(j, :), 56, 46), 'Colormap', gray);
        axis off;
        for i = 1:8
            dimen = eigenvector(:, 1:i*20);
            low_matrix = face_matrix(j, :) * dimen;
            face = low_matrix * dimen';
            face = reshape(face, 56, 46);
            subplot(3, 3, i+1);
            face_ = uint8(face * 5);
            imshow(face_, 'Colormap', gray);
            title(sprintf('d=%d', i*20));
            axis off;
        end
        pause;
    end
end
