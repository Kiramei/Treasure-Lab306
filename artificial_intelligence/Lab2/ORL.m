
y_axis_data = [];
for r = 10:10:160
    x_value = [];
    y_value = [];
    fprintf('当降维到%d时\n', r);
    k = 5;
    [train_face, train_label, test_face, test_label] = Loaddata(k);
    [data_train_new, data_mean, V_r] = PCA(train_face, r);
    num_train = size(data_train_new, 1);
    num_test = size(test_face, 1);
    temp_face = test_face - repmat(data_mean, num_test, 1);
    data_test_new = temp_face * V_r;
    true_num = 0;
    SVMModel = fitcecoc(data_train_new, train_label);
    result = predict(SVMModel, data_test_new);
    accuracy = sum(result == test_label) / numel(test_label) * 100;
    y_axis_data = [y_axis_data; accuracy * 100];
    fprintf('当每个人选择%d张照片进行训练时，The classify accuracy is: %.2f%%\n', k, accuracy);
end

x_axis_data = 10:10:160;
plot(x_axis_data, y_axis_data, 'b*--', 'LineWidth', 1, 'MarkerSize', 5);
for i = 1:length(x_axis_data)
    text(x_axis_data(i), y_axis_data(i) + 0.3, sprintf('%.1f', y_axis_data(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 7.5);
end

legend('8 ：2');
xlabel('降维到的维度');
ylabel('识别准确率');

function [newdata, meanVal] = Centralization(dataMat)
    [rows, cols] = size(dataMat);
    meanVal = mean(dataMat, 1);
    newdata = dataMat - repmat(meanVal, rows, 1);
end

function [train_face, train_label, test_face, test_label] = Loaddata(k)
    imgsize = 92 * 112;
    Number = 40;
    perphotos = 10;
    train_face = zeros(Number * k, imgsize);
    train_label = zeros(Number * k, 1);
    test_face = zeros(Number * perphotos, imgsize);
    test_label = zeros(Number * perphotos, 1);
    sample = randperm(perphotos);
    for i = 1:Number
        peopleID = i;
        for j = 1:perphotos
            imgpath = sprintf('%s/s%d/%d.pgm', './Dataset/ORL', peopleID, sample(j));
            image = imread(imgpath);
            [rows, cols] = size(image);
            imgVector = reshape(image, 1, rows * cols);
            if j <= k && i <= 40
                train_face((i - 1) * k + j, :) = imgVector;
                train_label((i - 1) * k + j) = peopleID;
            end
            test_face((i - 1) * (perphotos) + (j), :) = imgVector;
            test_label((i - 1) * (perphotos) + (j)) = peopleID;
        end
    end
end

function [final_data, meanVal, V_r] = PCA(data, r)
    dataMat = single(data);
    [A, meanVal] = Centralization(dataMat);
    covMat = A * A';
    [V, D] = eig(covMat);
    [D, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    V_r = V(:, 1:r);
    V_r = A' * V_r;
    for i = 1:r
        V_r(:, i) = V_r(:, i) / norm(V_r(:, i));
    end
    final_data = A * V_r;
end
